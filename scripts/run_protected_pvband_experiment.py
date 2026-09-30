"""Preregistered Linux/RTX 5090 source-only PV-band experiment."""
import argparse, gc, hashlib, json, math, os, platform, signal, subprocess, sys, traceback, uuid
from datetime import datetime, timezone
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from light_source import DifferentiableAbbeLitho, PixelatedLightSource, resist_image
from source_training import ProcessCorner, SourceDataset, SourceFitConfig, _cpu_state, _prepare, _printed

RASTER, PIXEL = 128, 4.0
SEEDS, DOSES, STEPS, CHECKPOINTS = (17,29,43,71,101), (0.98,1.,1.02), 200, (40,100,200)
CORNERS = tuple(ProcessCorner("nominal" if d == 1 else "d%g" % d, dose=d) for d in DOSES)
THRESHOLD, STEEPNESS, BINARY = 0.225, 50., 0.5
JITTER, SURROGATE_K, SURROGATE_W, RADIUS = 0.1, 20., 0.05, 2
MAX_BASIS, MAX_TOTAL, MAX_DEVICE = 512*1024**2, 2*1024**3, 512*1024**2
ARMS = ("A0","A4","A5")
FINAL_IDS = ("test_lines_diagonal_space","test_contacts_staggered_array","test_junction_asymmetric_cross")


def sha256_tensor(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path, obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def save_pt(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(obj,tmp); os.replace(tmp,path)


def rect(m,y0,y1,x0,x1):
    h,w=m.shape; y0,y1=max(0,y0),min(h,y1); x0,x1=max(0,x0),min(w,x1)
    if y0<y1 and x0<x1: m[y0:y1,x0:x1]=1


def calibration_layouts(size=RASTER):
    rows=[]
    def fresh(i,f,g):
        m=torch.zeros((size,size)); rows.append({"layout_id":i,"family":f,"geometry":g,"mask":m}); return m
    m=fresh("cal_lines_phase_shifted_ribbons","lines","four 36 nm clear ribbons with staggered clipped ends and mixed spacing")
    for i,x in enumerate((7,39,75,108)):
        for y in range(size):
            dx=((y//12)+3*i)%7-3; rect(m,y,y+1,x+dx,x+dx+9)
    m=fresh("cal_contacts_hexagonal_array","contacts","twelve 96 nm square contacts in staggered 4 by 3 array")
    for j,y in enumerate((9,41,73,105)):
        for x in (8+(16 if j%2 else 0),40+(16 if j%2 else 0),72+(16 if j%2 else 0)): rect(m,y,y+24,x,x+24)
    m=fresh("cal_junctions_asymmetric_y_tree","junctions","two asymmetric Y junctions with 40 nm trunks and 36 nm branches")
    for x,y in ((26,24),(85,73)):
        rect(m,y+14,y+62,x+18,x+28)
        for k in range(20):
            rect(m,y+14-k,y+15-k,x+18-k,x+28-k); rect(m,y+14-k,y+15-k,x+28+k,x+38+k)
    m=fresh("cal_junctions_offset_double_t","junctions","offset double-T network with 40 nm trunks and 36 nm capped crossbars")
    for spec in ((15,113,59,69),(31,41,28,100),(74,84,18,87),(102,112,69,108)): rect(m,*spec)
    return rows


def teacher_source(device):
    s=PixelatedLightSource(9,sigma_inner=.3,sigma_outer=.9); c,mask=s._coordinates,s._pupil_support
    x,y=c[...,0],c[...,1]
    a=-((x-.45).square()+(y-.225).square())/(2*.24**2)
    b=-((x+.225).square()+(y+.45).square())/(2*.27**2)
    with torch.no_grad(): s.logits.copy_(torch.where(mask,torch.logaddexp(a,b+math.log(.55)),s.logits))
    return s.to(device)


def simulator(device,seed):
    s=PixelatedLightSource(9,sigma_inner=.3,sigma_outer=.9)
    gen=torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad(): s.logits.add_(torch.randn(s.logits.shape,generator=gen)*JITTER*s._pupil_support)
    return DifferentiableAbbeLitho(s,numerical_aperture=1.35,wavelength_nm=193.,pixel_size_nm=PIXEL,
                                    source_chunk_size=8,cache_max_bytes=0).to(device)


def as_dataset(rows):
    return SourceDataset(torch.stack([r["mask"] for r in rows]),torch.stack([r["target"] for r in rows]),
                         tuple(r["layout_id"] for r in rows),PIXEL)


def generate_calibration(device):
    rows=calibration_layouts()
    teacher=DifferentiableAbbeLitho(teacher_source(device),numerical_aperture=1.35,wavelength_nm=193.,
                                    pixel_size_nm=PIXEL,source_chunk_size=8,cache_max_bytes=0).to(device)
    with torch.no_grad():
        for r in rows:
            printed=resist_image(teacher(r["mask"].to(device)),dose=1.,threshold=THRESHOLD,steepness=STEEPNESS)
            r["target"]=(printed>=BINARY).float().cpu()[0]
    return teacher,as_dataset(rows),rows


def load_datasets(path):
    p=torch.load(path,map_location="cpu",weights_only=True)
    if not isinstance(p,dict) or any(k not in p for k in ("fit","calibration","final_test")): raise ValueError("dataset file needs fit/calibration/final_test")
    ds={k:SourceDataset(**p[k]) for k in ("fit","calibration","final_test")}
    if ds["final_test"].layout_ids!=FINAL_IDS: raise ValueError("unexpected final-test IDs/order")
    for k,d in ds.items():
        if d.pixel_size_nm!=PIXEL or tuple(d.targets.shape[-2:])!=(RASTER,RASTER): raise ValueError("unexpected %s raster/pixel size"%k)
    return ds


def novelty_check(new,old):
    old_m={sha256_tensor(x) for d in old.values() for x in d.masks}
    old_t={sha256_tensor(x) for d in old.values() for x in d.targets}
    mh=[sha256_tensor(x) for x in new.masks]; th=[sha256_tensor(x) for x in new.targets]
    if len(set(mh))!=len(mh) or len(set(th))!=len(th): raise ValueError("duplicate new calibration masks/targets")
    dm=[new.layout_ids[i] for i,x in enumerate(mh) if x in old_m]
    dt=[new.layout_ids[i] for i,x in enumerate(th) if x in old_t]
    if dm or dt: raise ValueError("new calibration duplicates prior data: masks=%s targets=%s"%(dm,dt))
    return {"previous_layouts":sum(len(d.layout_ids) for d in old.values()),"previous_mask_hashes":len(old_m),
            "previous_target_hashes":len(old_t),"duplicate_masks":[],"duplicate_targets":[]}


def preflight(new,old):
    frac=[float(x.mean()) for x in new.targets]
    bad=[(new.layout_ids[i],x) for i,x in enumerate(frac) if not .01<=x<=.99]
    if bad: raise ValueError("targets outside 1%-99%; correct only for degeneracy before fitting: %r"%bad)
    return {"targets":[{"layout_id":new.layout_ids[i],"mask_sha256":sha256_tensor(new.masks[i]),
                        "target_sha256":sha256_tensor(new.targets[i]),"positive_pixels":int(new.targets[i].sum()),
                        "positive_fraction":frac[i]} for i in range(len(frac))],
            "novelty":novelty_check(new,old),"design_corrections":[]}


def contour_mask(target,radius=RADIUS):
    t=torch.as_tensor(target).detach()
    if t.ndim==2: t=t[None,None]
    elif t.ndim==3: t=t.unsqueeze(0)
    if t.ndim!=4 or tuple(t.shape[:2])!=(1,1): raise ValueError("target must be one (1,1,H,W) image")
    t=t.float()
    if not torch.isfinite(t).all() or not torch.all((t==0)|(t==1)): raise ValueError("target must be finite binary")
    if not t.any() or t.all(): raise ValueError("contour requires foreground and background")
    if isinstance(radius,bool) or not isinstance(radius,int) or radius<0: raise ValueError("bad radius")
    pad=F.pad(t,(1,1,1,1),value=0)
    dil=F.max_pool2d(pad,3,stride=1)
    ero=-F.max_pool2d(F.pad(-t,(1,1,1,1),value=0),3,stride=1)
    contour=(dil>0)&(ero<1); perimeter=int(contour.sum())
    roi=F.max_pool2d(contour.float(),2*radius+1,stride=1,padding=radius)>0 if radius else contour
    if perimeter<1: raise ValueError("empty target contour")
    return roi,perimeter


def balanced_mse(printed,target):
    if printed.ndim!=5: raise ValueError("printed shape must be (corner,batch,1,H,W)")
    target=target.to(device=printed.device,dtype=printed.dtype); fg=target>.5; bg=~fg
    if not fg.any() or not bg.any(): raise ValueError("balanced MSE needs both classes")
    worst=(printed-target.unsqueeze(0)).square().amax(dim=0)
    return .5*worst[fg].mean()+.5*worst[bg].mean()


def local_band(printed,target,roi=None,perimeter=None):
    if roi is None or perimeter is None: roi,perimeter=contour_mask(target)
    if perimeter<=0: raise ValueError("bad contour perimeter")
    q=torch.sigmoid(SURROGATE_K*(printed-.5)); b=q.amax(dim=0)-q.amin(dim=0)
    return (b*roi.to(device=printed.device,dtype=printed.dtype)).sum()/float(perimeter)


def objective_parts(printed,target,arm,roi=None,perimeter=None):
    if arm not in ARMS: raise ValueError("unknown arm")
    target=target.to(device=printed.device,dtype=printed.dtype)
    f=(printed-target.unsqueeze(0)).square().mean()
    e=(printed.amax(dim=0)-printed.amin(dim=0)).square().mean()
    b=balanced_mse(printed,target); s=local_band(printed,target,roi,perimeter)
    loss=f+.5*e if arm=="A0" else b
    if arm=="A5": loss=loss+SURROGATE_W*s
    return {"fidelity_mse":f,"envelope_mse":e,"balanced_worst_mse":b,"localized_surrogate_band":s,"objective":loss}


def cfg(seed):
    return SourceFitConfig(steps=STEPS,learning_rate=.01,threshold=THRESHOLD,steepness=STEEPNESS,
                           binary_threshold=BINARY,max_basis_bytes=MAX_BASIS,max_total_basis_bytes=MAX_TOTAL,
                           max_device_basis_bytes=MAX_DEVICE,seed=seed,verify_basis=True)


def geoms(ds): return [contour_mask(x) for x in ds.targets]


def resident(sim,ds,config):
    b,p=_prepare(sim,ds,(0.,),config)
    return [{0.:x[0.].to(sim.source.logits.device)} for x in b],p


def evaluate(sim,ds,bases,arm,geometry):
    records=[]; nominal=1
    for i,b in enumerate(bases):
        printed=_printed(sim,b,CORNERS,cfg(0)); target=ds.targets[i:i+1].to(printed.device)
        if not torch.isfinite(printed).all(): raise FloatingPointError("nonfinite print")
        binary=printed>=BINARY; l2=(binary!=target.bool().unsqueeze(0)).reshape(3,-1).sum(1)
        pos=int(target.sum()); pc=[]
        for j,c in enumerate(CORNERS):
            pred,truth=binary[j],target.bool()
            fn=int((~pred&truth).sum())
            pc.append({"corner":{"name":c.name,"dose":c.dose},"predicted_positive_pixels":int(pred.sum()),
                       "target_positive_pixels":pos,"false_positive_pixels":int((pred&~truth).sum()),
                       "false_negative_pixels":fn,"recall":float(pos-fn)/pos if pos else None})
        band=int((binary.any(0)!=binary.all(0)).sum()); roi,per=geometry[i]
        parts=objective_parts(printed,target,arm,roi.to(printed.device),per)
        records.append({"layout_id":ds.layout_ids[i],"target_positive_pixels":pos,"L2_pixels":int(l2[nominal]),
                        "L2_worst_dose_pixels":int(l2.max()),"band_pixels":band,"per_corner_binary_metrics":pc,
                        **{k:float(v.detach()) for k,v in parts.items()}})
    keys=("L2_pixels","L2_worst_dose_pixels","band_pixels","fidelity_mse","envelope_mse",
          "balanced_worst_mse","localized_surrogate_band","objective")
    return {"mean":{k:sum(r[k] for r in records)/len(records) for k in keys},"per_layout":records}


def final_test_arms(sel):
    if not isinstance(sel,dict) or sel.get("selected_arm") not in ("A4","A5"): return []
    return ["A0",sel["selected_arm"]] if sel["selected_arm"] in sel.get("eligible_arms",[sel["selected_arm"]]) else []


def choose_arm(runs):
    if any(len(runs.get(a,[]))!=len(SEEDS) for a in ARMS):
        return {"selected_arm":None,"eligible_arms":[],"candidate_checks":{},"final_gate_arms":[]}
    base=[r["calibration"] for r in runs["A0"]]
    bb=[r["mean"]["band_pixels"] for r in base]; bn=[r["mean"]["L2_pixels"] for r in base]
    bw=[r["mean"]["L2_worst_dose_pixels"] for r in base]; checks={}
    for arm in ("A4","A5"):
        rs=[r["calibration"] for r in runs[arm]]
        band=[r["mean"]["band_pixels"] for r in rs]; nom=[r["mean"]["L2_pixels"] for r in rs]
        worst=[r["mean"]["L2_worst_dose_pixels"] for r in rs]
        zero=[x["layout_id"] for r in rs for x in r["per_layout"] if x["target_positive_pixels"]>0
              and x["per_corner_binary_metrics"][1]["predicted_positive_pixels"]==0]
        bm=sum(bb)/len(SEEDS); cm=sum(band)/len(SEEDS); nm=sum(nom)/len(SEEDS); wm=sum(worst)/len(SEEDS)
        checks[arm]={"pv_band_mean_reduction_at_least_10pct":bm>0 and cm<=.9*bm,
                     "nominal_L2_mean_at_most_5pct_worse":nm<=1.05*sum(bn)/len(SEEDS),
                     "worst_dose_L2_mean_at_most_5pct_worse":wm<=1.05*sum(bw)/len(SEEDS),
                     "pv_band_improves_in_each_seed":all(band[i]<bb[i] for i in range(len(SEEDS))),
                     "no_positive_layout_with_zero_nominal_print":not zero,"zero_print_layout_ids":sorted(set(zero)),
                     "calibration_means":{"band_pixels":cm,"L2_pixels":nm,"L2_worst_dose_pixels":wm},
                     "qualified":False}
        checks[arm]["qualified"]=all(checks[arm][k] for k in (
          "pv_band_mean_reduction_at_least_10pct","nominal_L2_mean_at_most_5pct_worse",
          "worst_dose_L2_mean_at_most_5pct_worse","pv_band_improves_in_each_seed",
          "no_positive_layout_with_zero_nominal_print"))
    eligible=[a for a in ("A4","A5") if checks[a]["qualified"]]
    selected=min(eligible,key=lambda a:(checks[a]["calibration_means"]["band_pixels"],
                                        checks[a]["calibration_means"]["L2_pixels"],a)) if eligible else None
    out={"selected_arm":selected,"eligible_arms":eligible,"candidate_checks":checks,
         "final_gate_arms":[],"selection_rule":"calibration only: PV-band <=0.90*A0 and lower each seed, nominal/worst L2 <=1.05*A0, no zero nominal positive layout; pick lower band"}
    out["final_gate_arms"]=final_test_arms(out); return out


def gpu_stats(device):
    temp=None
    try:
        p=subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"],
                         check=True,capture_output=True,text=True,timeout=5); temp=int(p.stdout.splitlines()[0])
    except (OSError,subprocess.SubprocessError,ValueError,IndexError): pass
    return {"temperature_c":temp,"memory_allocated_bytes":int(torch.cuda.memory_allocated(device)),
            "memory_reserved_bytes":int(torch.cuda.memory_reserved(device)),
            "peak_memory_allocated_bytes":int(torch.cuda.max_memory_allocated(device))}


def fit_seed(sim,train,cal,arm,seed,run_dir,record,device,save_report):
    config=cfg(seed); tb,tp=resident(sim,train,config); cb,cp=resident(sim,cal,config)
    tg,cg=geoms(train),geoms(cal); opt=torch.optim.Adam([sim.source.logits],lr=.01); hist=[]; diags=[]
    record.update({"seed":seed,"status":"running","initial_logits_sha256":sha256_tensor(sim.source.logits),
                   "step_history":hist,"diagnostics":diags,"basis_verification":tp+cp})
    try:
        for step in range(1,STEPS+1):
            opt.zero_grad(set_to_none=True); parts={k:0. for k in ("fidelity_mse","envelope_mse","balanced_worst_mse","localized_surrogate_band","objective")}
            for i,b in enumerate(tb):
                printed=_printed(sim,b,CORNERS,config); target=train.targets[i:i+1].to(device); roi,per=tg[i]
                terms=objective_parts(printed,target,arm,roi.to(device),per); loss=terms["objective"]/len(train.masks)
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite loss")
                loss.backward()
                for k,v in terms.items(): parts[k]+=float(v.detach())/len(train.masks)
            if sim.source.logits.grad is None or not torch.isfinite(sim.source.logits.grad).all(): raise FloatingPointError("bad gradient")
            opt.step()
            if not torch.isfinite(sim.source.logits).all(): raise FloatingPointError("nonfinite logits")
            hist.append({"step":step,**parts})
            if step%10==0:
                save_pt({"arm":arm,"seed":seed,"step":step,"source_state":_cpu_state(sim.source),"step_history":hist},
                        run_dir/"checkpoints"/arm/("seed_%d"%seed)/"latest.pt")
                record["steps_completed"]=step; save_report()
            if step in CHECKPOINTS:
                with torch.no_grad(): fm=evaluate(sim,train,tb,arm,tg); cm=evaluate(sim,cal,cb,arm,cg)
                stats=gpu_stats(device); path=run_dir/"checkpoints"/arm/("seed_%d"%seed)/("step_%03d.pt"%step)
                save_pt({"arm":arm,"seed":seed,"step":step,"source_state":_cpu_state(sim.source),
                         "step_history":hist,"gpu":stats},path)
                flux=float(sim.source.weight_map().sum().detach())
                if not math.isfinite(flux) or abs(flux-1.0)>1e-6: raise FloatingPointError("source flux left unit normalization")
                diags.append({"step":step,"fit":fm,"calibration":cm,"loss_components":parts,
                              "source_flux_sum":flux,"gpu":stats,"checkpoint":str(path)})
                record.update({"steps_completed":step,"after":{"fit":fm,"calibration":cm}})
                save_report()
                print(json.dumps({"event":"checkpoint","arm":arm,"seed":seed,"step":step,
                                  "calibration_band":cm["mean"]["band_pixels"],"gpu":stats}),flush=True)
        record["status"]="complete"; return record
    except BaseException:
        record["status"]="interrupted_or_failed"; record["steps_completed"]=len(hist)
        raise
    finally:
        for group in (tb,cb):
            for b in group: b[0.].cpu()


def require_device(expected):
    if platform.system()!="Linux" or not torch.cuda.is_available(): raise RuntimeError("requires Linux and CUDA")
    dev=torch.device("cuda:0"); name=torch.cuda.get_device_name(dev)
    if expected!="NVIDIA GeForce RTX 5090" or name!=expected: raise RuntimeError("expected RTX 5090, found "+name)
    torch.cuda.synchronize(dev); return dev,name


def run(args):
    root=Path(args.output_root)
    if not root.is_absolute(): raise ValueError("--output-root must be absolute")
    device,gpu=require_device(args.expected_gpu); previous=load_datasets(args.dataset_file)
    folder=root/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]); folder.mkdir(parents=True)
    report_path=folder/"experiment.json"
    report={"schema_version":1,"status":"preflight_running","created_utc":datetime.now(timezone.utc).isoformat(),
      "result_directory":str(folder),"system":{"python":sys.version,"platform":platform.platform(),"torch":str(torch.__version__),
      "cuda":torch.version.cuda,"gpu":gpu,"device":str(device)},
      "protocol":{"dataset_file":str(Path(args.dataset_file).resolve()),"raster":[RASTER,RASTER],"pixel_size_nm":PIXEL,
      "threshold":THRESHOLD,"steepness":STEEPNESS,"binary_threshold":BINARY,"doses":list(DOSES),"focus":"none",
      "seeds":list(SEEDS),"steps":STEPS,"optimizer":{"name":"Adam","learning_rate":.01},"jitter_std":JITTER,
      "checkpoint_steps":list(CHECKPOINTS),"arms":{"A0":"fidelity MSE + 0.5 envelope squared",
      "A4":"class-balanced per-pixel worst-corner MSE","A5":"A4 + 0.05 contour-localized surrogate PV-band"},
      "A5_definition":{"q":"sigmoid(20*(printed-0.5))","ROI":"fixed radius-2 contour dilation",
      "normalization":"fixed contour pixel count/perimeter"},"final_rule":"only after calibration gate evaluate A0 plus frozen candidate",
      "memory_budgets":{"per_basis":MAX_BASIS,"resident_total":MAX_TOTAL,"per_layout_device":MAX_DEVICE,
                        "gpu_capacity_reference":32*1024**3}},
      "previous_splits":{k:list(v.layout_ids) for k,v in previous.items()},
      "previous_dataset_hashes":{k:[{"layout_id":v.layout_ids[i],"mask_sha256":sha256_tensor(v.masks[i]),
        "target_sha256":sha256_tensor(v.targets[i])} for i in range(len(v.layout_ids))] for k,v in previous.items()},
      "preflight":None,"teacher_reference":None,"initial_logit_hashes":{},"arms_run":{},
      "selection":None,"final_test":{"status":"closed"}}
    write_json(report_path,report)
    try:
        teacher,cal,rows=generate_calibration(device); train=previous["fit"]
        report["preflight"]=preflight(cal,previous)
        report["preflight"]["geometry"]=[{k:r[k] for k in ("layout_id","family","geometry")} for r in rows]
        tr={}
        for name,ds in (("fit",train),("calibration",cal)):
            b,parity=_prepare(teacher,ds,(0.,),cfg(SEEDS[0]))
            with torch.no_grad(): metrics=evaluate(teacher,ds,[{0.:x[0.]} for x in b],"A0",geoms(ds))
            tr[name]={"metrics":metrics,"basis_verification":parity}
            for x in b: x[0.].cpu()
        report["teacher_reference"]=tr
        _,w=teacher.source.distribution(); per=int(w.numel()*RASTER*RASTER*4)
        total=per*(len(train.masks)+len(cal.masks))
        if per>MAX_BASIS or total>MAX_TOTAL: raise MemoryError("basis budget exceeded")
        report["basis_estimate_bytes"]={"per_layout":per,"resident_fit_calibration":total}
        report["teacher_source_weights"]=teacher.source.weight_map().detach().cpu().tolist(); del teacher
        report["status"]="preflight_passed"; write_json(report_path,report)
        report["initial_source_flux_sums"]={}
        for seed in SEEDS:
            hashes=[]; fluxes=[]
            for arm in ARMS:
                s=simulator(device,seed); hashes.append(sha256_tensor(s.source.logits))
                fluxes.append(float(s.source.weight_map().sum().detach())); del s
            if len(set(hashes))!=1: raise AssertionError("initial logits mismatch")
            if any(not math.isfinite(v) or abs(v-1.0)>1e-6 for v in fluxes): raise AssertionError("source flux is not normalized")
            report["initial_logit_hashes"][str(seed)]={a:hashes[0] for a in ARMS}
            report["initial_source_flux_sums"][str(seed)]={a:fluxes[0] for a in ARMS}
        report["status"]="fitting"; write_json(report_path,report)
        for arm in ARMS:
            report["arms_run"][arm]=[]
            for seed in SEEDS:
                sim=simulator(device,seed); rec={}; report["arms_run"][arm].append(rec)
                fit_seed(sim,train,cal,arm,seed,folder,rec,device,lambda:write_json(report_path,report))
                write_json(report_path,report); del sim; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
        selection_input={arm:[{"calibration":rec["after"]["calibration"]} for rec in records]
                         for arm,records in report["arms_run"].items()}
        report["selection"]=choose_arm(selection_input); report["status"]="calibration_gate_complete"
        write_json(report_path,report); gate=final_test_arms(report["selection"])
        if not gate:
            report["final_test"]={"status":"not_run_no_qualifying_candidate"}; report["status"]="done_no_qualifying_candidate"
            write_json(report_path,report); return report
        # Final predictions are first produced after the frozen calibration gate.
        final=previous["final_test"]; fg=geoms(final); results=[]
        for seed in SEEDS:
            for arm in gate:
                sim=simulator(device,seed); rec=next(x for x in report["arms_run"][arm] if x["seed"]==seed)
                cp=next(d["checkpoint"] for d in rec["diagnostics"] if d["step"]==STEPS)
                saved=torch.load(cp,map_location="cpu",weights_only=True)
                sim.source.load_state_dict(saved["source_state"]); b,parity=resident(sim,final,cfg(seed))
                with torch.no_grad(): metrics=evaluate(sim,final,b,arm,fg)
                for x in b: x[0.].cpu()
                results.append({"seed":seed,"arm":arm,"metrics":metrics,"basis_verification":parity,"gpu":gpu_stats(device)})
                del sim; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
                report["final_test"]={"status":"evaluating_after_calibration_freeze","arms":gate,
                  "frozen_candidate":report["selection"]["selected_arm"],"runs":results}; write_json(report_path,report)
        report["final_test"]["status"]="evaluated_after_calibration_freeze"; report["status"]="done"; write_json(report_path,report)
        return report
    except BaseException as exc:
        report["status"]="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed"
        if report["selection"] is None: report["final_test"]={"status":"closed"}
        report["failure"]={"type":type(exc).__name__,"message":str(exc)}
        write_json(report_path,report); (folder/"failure.log").write_text(traceback.format_exc(),encoding="utf-8"); raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--expected-gpu",required=True)
    args=parser.parse_args(argv)
    if args.expected_gpu!="NVIDIA GeForce RTX 5090": parser.error("--expected-gpu must be NVIDIA GeForce RTX 5090")
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    try: report=run(args)
    except KeyboardInterrupt:
        print("Interrupted; incremental report and CPU checkpoints preserved.",flush=True); return 130
    print("Status:",report["status"]); print("Report:",Path(report["result_directory"])/"experiment.json")
    print("Selection:",(report.get("selection") or {}).get("selected_arm")); return 0


if __name__=="__main__": raise SystemExit(main())
