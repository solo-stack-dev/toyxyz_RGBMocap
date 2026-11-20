"""Exercise native LieTorch and DroidCalib GPU (AMD ROCm/HIP) operations and scatter."""
from pathlib import Path
import argparse,json,sys


def verify(folder):
    sys.path.insert(0,str(folder))
    import torch,lietorch,torch_scatter,droid_backends_intr
    if not torch.cuda.is_available() or not getattr(torch.version,'hip',None):raise RuntimeError('PromptHMR camera support requires the AMD ROCm GPU runtime.')
    torch.set_num_threads(2);torch.manual_seed(5)
    x=torch.tensor([[.1,.2,.3,.01,.02,.03]],device='cuda',requires_grad=True)
    transform=lietorch.SE3.exp(x);points=torch.tensor([[1.,2.,3.]],device='cuda')
    y=transform.act(points);y.sum().backward()
    if not torch.isfinite(x.grad).all() or not torch.allclose((transform.inv()*transform).matrix(),torch.eye(4,device='cuda')[None],atol=1e-5):raise ValueError('LieTorch GPU verification failed.')
    # The GPU kernels must match LieTorch's own CPU implementation, values and gradients.
    worst=0.
    for group,dim in ((lietorch.SE3,6),(lietorch.Sim3,7),(lietorch.SO3,3)):
        tangent=torch.randn(64,dim,dtype=torch.float64)*.5;cloud=torch.randn(64,3,dtype=torch.float64)
        results=[]
        for device in ('cpu','cuda'):
            a=tangent.to(device,copy=True).requires_grad_();p=cloud.to(device,copy=True).requires_grad_()
            g=group.exp(a);out=torch.cat([(g*g.inv()*g).log(),g.act(p),g.adjT(a).reshape(64,-1)],-1)
            out.square().sum().backward()
            results.append([t.detach().cpu() for t in (out,a.grad,p.grad)])
        for c,g in zip(*results):worst=max(worst,float((c-g).abs().max()))
    if worst>1e-9:raise ValueError('LieTorch GPU/CPU mismatch: %.3g'%worst)
    result=torch_scatter.scatter_sum(torch.tensor([1.,2.,3.],device='cuda'),torch.tensor([0,1,0],device='cuda'))
    mean=torch_scatter.scatter_mean(torch.tensor([[1.,2.],[3.,4.],[5.,6.]],device='cuda')[None],torch.tensor([0,2,0],device='cuda'),dim=1,dim_size=4)
    if result.tolist()!=[4.,2.] or mean.tolist()!=[[[3.,4.],[0.,0.],[3.,4.],[0.,0.]]]:raise ValueError('Scatter verification failed.')
    poses=lietorch.SE3.Identity(2,device='cuda').data;disps=torch.ones((2,8,8),device='cuda')
    intr=torch.tensor([[10.,10.,4.,4.]]*2,device='cuda');ii=torch.tensor([0],device='cuda');jj=torch.tensor([1],device='cuda')
    distance=droid_backends_intr.frame_distance(poses,disps,intr[0],ii,jj,.3,2)
    if not torch.isfinite(distance).all() or distance.abs().max()>.001:raise ValueError('DroidCalib GPU verification failed.')
    # Correlation lookup: bilinear samples of each pixel's correlation map.
    n,h,w,r=2,6,7,3;volume=torch.randn(n,h,w,9,10,device='cuda');coords=torch.rand(n,2,h,w,device='cuda')*torch.tensor([10.,9.],device='cuda').view(1,2,1,1)-.5
    corr,=droid_backends_intr.corr_index_forward(volume,coords,r)
    offsets=torch.arange(-r,r+1,device='cuda',dtype=torch.float32)
    sx=coords[:,0].reshape(-1,1,1)+offsets.view(-1,1);sy=coords[:,1].reshape(-1,1,1)+offsets.view(1,-1)
    grid=torch.stack([sx.expand(-1,-1,2*r+1)*2/9-1,sy.expand(-1,2*r+1,-1)*2/8-1],-1)
    reference=torch.nn.functional.grid_sample(volume.reshape(-1,1,9,10),grid,align_corners=True,padding_mode='zeros')
    reference=reference.reshape(n,h,w,2*r+1,2*r+1).permute(0,3,4,1,2)
    corr_error=float((corr-reference).abs().max())
    if corr_error>1e-4:raise ValueError('DroidCalib correlation verification failed: %.3g'%corr_error)
    torch.cuda.synchronize()
    return dict(compute_capability=list(torch.cuda.get_device_capability()),gpu_arch=torch.cuda.get_device_properties(0).gcnArchName.split(':')[0],
                torch=torch.__version__,hip=torch.version.hip,lietorch_gpu=True,lietorch_cpu_gpu_error=worst,scatter=True,droid_gpu=True,droid_corr_error=corr_error)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=verify(args.folder);args.output.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result),flush=True)
