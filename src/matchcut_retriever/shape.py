"""Numpy/OpenCV geometry and shape matching primitives."""
from __future__ import annotations

from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class ObjectGeometry:
    frame_w: int; frame_h: int; cx: float; cy: float; area: float
    bbox: tuple[int, int, int, int]; angle: float; eccentricity: float
    @property
    def area_frac(self) -> float: return self.area / (self.frame_w * self.frame_h)
    @property
    def bbox_norm(self) -> tuple[float, float, float, float]:
        x0,y0,x1,y1=self.bbox
        return (x0/self.frame_w,y0/self.frame_h,x1/self.frame_w,y1/self.frame_h)


def object_geometry(mask: np.ndarray) -> ObjectGeometry:
    m=np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not m.any(): raise ValueError("mask must be a non-empty 2-D array")
    h,w=m.shape; yy,xx=np.nonzero(m); area=float(len(xx))
    cx=float(xx.mean()); cy=float(yy.mean())
    x0,x1=int(xx.min()),int(xx.max()+1); y0,y1=int(yy.min()),int(yy.max()+1)
    dx=xx-cx; dy=yy-cy
    cov=np.array([[np.mean(dx*dx),np.mean(dx*dy)],[np.mean(dx*dy),np.mean(dy*dy)]])
    vals,vecs=np.linalg.eigh(cov); order=np.argsort(vals)[::-1]; vals=vals[order]; v=vecs[:,order[0]]
    angle=float(math.atan2(v[1],v[0])); angle=((angle+math.pi/2)%math.pi)-math.pi/2
    # represent the upper endpoint as +pi/2 when numerically exactly there
    if angle <= -math.pi/2 + 1e-12: angle=math.pi/2
    ecc=0.0 if vals[0] <= 1e-12 else float(math.sqrt(max(0.,1.-vals[1]/vals[0])))
    return ObjectGeometry(w,h,cx,cy,area,(x0,y0,x1,y1),angle,ecc)


FINGERPRINT_SIZE=32
FINGERPRINT_DIM=FINGERPRINT_SIZE*FINGERPRINT_SIZE

def shape_fingerprint(mask: np.ndarray) -> np.ndarray:
    m=np.asarray(mask,dtype=np.uint8)
    g=object_geometry(m)
    yy,xx=np.nonzero(m); scale=max(math.sqrt(g.area),1e-6)
    # Coordinates are centroid/area normalized, retaining aspect ratio.
    lim=4.0; q=((xx-g.cx)/scale + lim)/(2*lim)*(FINGERPRINT_SIZE-1)
    r=((yy-g.cy)/scale + lim)/(2*lim)*(FINGERPRINT_SIZE-1)
    img=np.zeros((FINGERPRINT_SIZE,FINGERPRINT_SIZE),np.float32)
    xi=np.rint(q).astype(int); yi=np.rint(r).astype(int)
    ok=(xi>=0)&(xi<FINGERPRINT_SIZE)&(yi>=0)&(yi<FINGERPRINT_SIZE)
    img[yi[ok],xi[ok]]=1.0
    img=cv2.GaussianBlur(img,(5,5),0)
    v=img.ravel().astype(np.float32); n=float(np.linalg.norm(v));
    if n: v/=n
    return v

def fingerprint_matrix(masks) -> np.ndarray:
    return np.asarray([shape_fingerprint(m) for m in masks],dtype=np.float32).reshape((-1,FINGERPRINT_DIM))


@dataclass(frozen=True)
class Similarity:
    scale: float=1.0; angle: float=0.0; tx: float=0.0; ty: float=0.0
    def matrix(self, canvas_wh=None) -> np.ndarray:
        cs, sn=math.cos(self.angle),math.sin(self.angle)
        a=self.scale*np.array([[cs,-sn],[sn,cs]])
        if canvas_wh is None:
            return np.array([[a[0,0],a[0,1],self.tx],[a[1,0],a[1,1],self.ty]],np.float64)
        w,h=canvas_wh; c=np.array([w/2.,h/2.])
        t=c+np.array([self.tx,self.ty])-a@c
        return np.array([[a[0,0],a[0,1],t[0]],[a[1,0],a[1,1],t[1]]],np.float64)
    def apply(self, points, canvas_wh=None) -> np.ndarray:
        p=np.asarray(points,float); 
        if canvas_wh is None:
            a=self.scale*np.array([[math.cos(self.angle),-math.sin(self.angle)],[math.sin(self.angle),math.cos(self.angle)]])
            return p@a.T+np.array([self.tx,self.ty])
        w,h=canvas_wh; c=np.array([w/2.,h/2.]); a=self.scale*np.array([[math.cos(self.angle),-math.sin(self.angle)],[math.sin(self.angle),math.cos(self.angle)]])
        return (p-c)@a.T+c+np.array([self.tx,self.ty])


def cover_fit_matrix(cand_wh, query_wh) -> np.ndarray:
    wc,hc=map(float,cand_wh); wq,hq=map(float,query_wh); s=max(wq/wc,hq/hc)
    return np.array([[s,0.,(wq-s*wc)/2],[0.,s,(hq-s*hc)/2]],float)

@dataclass(frozen=True)
class HardLimits:
    min_scale: float=.8; max_scale: float=1.25; max_translate_frac: float=.10; max_rot_deg: float=6.0
@dataclass(frozen=True)
class SoftParams:
    """Controls the unconstrained shape match.

    ``scale_norm``, ``translate_norm`` and ``rotation_norm_deg`` are the
    characteristic scales used by :func:`_cost`: each corresponding
    displacement is divided by its constant and the three absolute terms are
    added.  Thus ``cost`` is dimensionless and zero for the identity
    transform; the constants are deliberately parameters so callers can tune
    the penalty without changing matching semantics.
    """
    allow_rotation: bool=True; ecc_threshold: float=.3; cost_weight: float=.1
    scale_norm: float=.25; translate_norm: float=.25; rotation_norm_deg: float=15.0
@dataclass
class MatchResult:
    iou: float; cost: float; score: float; transform: Similarity; within_limits: bool

def _closed(q,c,params):
    qg,cg=object_geometry(q),object_geometry(c); hq,wq=q.shape
    cf=cover_fit_matrix((c.shape[1],c.shape[0]),(wq,hq)); cp=np.array([cg.cx,cg.cy]); cp=cf[:,:2]@cp+cf[:,2]; qp=np.array([qg.cx,qg.cy]); center=np.array([wq/2,hq/2])
    s=math.sqrt(qg.area/(cg.area*(cf[0,0]**2)))
    a=0.0
    if params.allow_rotation and qg.eccentricity>params.ecc_threshold and cg.eccentricity>params.ecc_threshold:
        a=qg.angle-cg.angle
        a=(a+math.pi/2)%math.pi-math.pi/2
    R=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
    t=qp-(center+s*R@(cp-center))
    return Similarity(s,a,float(t[0]),float(t[1])),cf

def _warp_iou(q,c,sim,cf):
    h,w=q.shape; M=sim.matrix((w,h)); A=np.vstack([cf,[0,0,1]]); B=np.vstack([M,[0,0,1]])@A; B=B[:2]
    scale=min(1.,160/max(h,w)); ds=(max(1,int(round(w*scale))),max(1,int(round(h*scale))))
    qd=cv2.resize(q.astype(np.uint8),ds,interpolation=cv2.INTER_NEAREST)
    Bs=B.copy(); Bs[:,:2]*=scale; Bs[:,2]*=scale
    wd=cv2.warpAffine(c.astype(np.uint8),Bs,ds,flags=cv2.INTER_NEAREST,borderValue=0)
    inter=np.count_nonzero(qd&wd); union=np.count_nonzero(qd|wd)
    return float(inter/union) if union else 1.0

def _cost(sim,wh,params=SoftParams()):
    w,h=wh; tr=math.hypot(sim.tx,sim.ty)/(params.translate_norm*w+1e-9)
    return float(abs(math.log(max(sim.scale,1e-9)))/params.scale_norm + abs(sim.angle)/(math.radians(params.rotation_norm_deg)) + tr)

def soft_match(q_mask,c_mask,params=SoftParams()):
    sim,cf=_closed(q_mask,c_mask,params)
    # A principal axis is unoriented: an elongated object can match with
    # either endpoint of its axis corresponding to the query endpoint.  Try
    # the 180-degree alternative as well, retaining the higher-scoring one.
    candidates=[sim]
    qg,cg=object_geometry(q_mask),object_geometry(c_mask)
    if params.allow_rotation and qg.eccentricity>params.ecc_threshold and cg.eccentricity>params.ecc_threshold:
        h,w=q_mask.shape; center=np.array([w/2.,h/2.]); qp=np.array([qg.cx,qg.cy]); cp=cf[:,:2]@np.array([cg.cx,cg.cy])+cf[:,2]
        for a in (sim.angle+math.pi, sim.angle-math.pi):
            R=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
            t=qp-(center+sim.scale*R@(cp-center))
            candidates.append(Similarity(sim.scale,a,float(t[0]),float(t[1])))
    scored=[]
    for candidate in candidates:
        i=_warp_iou(q_mask,c_mask,candidate,cf)
        scored.append((i-params.cost_weight*_cost(candidate,(q_mask.shape[1],q_mask.shape[0]),params),i,candidate))
    # Axis direction is a shape alignment choice; select the best overlap,
    # using the cost only to break an overlap tie.  The selected transform's
    # cost is then applied once to form the soft score.
    _,iou,sim=max(scored,key=lambda x:(x[1],x[0]))
    cost=_cost(sim,(q_mask.shape[1],q_mask.shape[0]),params)
    lim=HardLimits(); within=(lim.min_scale<=sim.scale<=lim.max_scale and math.hypot(sim.tx,sim.ty)<=lim.max_translate_frac*q_mask.shape[1] and abs(sim.angle)<=math.radians(lim.max_rot_deg))
    return MatchResult(iou,cost,iou-params.cost_weight*cost,sim,within)

def hard_match(q_mask,c_mask,limits=HardLimits()):
    base,cf=_closed(q_mask,c_mask,SoftParams()); wq=q_mask.shape[1]; T=limits.max_translate_frac*wq; ar=math.radians(limits.max_rot_deg)
    within=(limits.min_scale<=base.scale<=limits.max_scale and math.hypot(base.tx,base.ty)<=T and abs(base.angle)<=ar)
    s=min(max(base.scale,limits.min_scale),limits.max_scale); a=min(max(base.angle,-ar),ar); t=np.array([base.tx,base.ty]);
    if np.linalg.norm(t)>T: t=t*T/np.linalg.norm(t)
    best=Similarity(s,a,float(t[0]),float(t[1])); biou=_warp_iou(q_mask,c_mask,best,cf)
    # small local refinement, preserving limits
    eps=.01*T
    direction=t/np.linalg.norm(t) if np.linalg.norm(t)>1e-9 else np.array([1.,0.])
    offsets=(-eps*direction, np.zeros(2), eps*direction)
    for ds in (-.01,0,.01):
      for da in (-math.radians(.5),0,math.radians(.5)):
       for offset in offsets:
        ss=min(max(s+ds,limits.min_scale),limits.max_scale); aa=min(max(a+da,-ar),ar); tt=t+offset
        if np.linalg.norm(tt)>T: continue
        z=Similarity(ss,aa,float(tt[0]),float(tt[1])); io=_warp_iou(q_mask,c_mask,z,cf)
        if io>biou: best,biou=z,io
    return MatchResult(biou,_cost(best,(wq,q_mask.shape[0])),biou, best,within)

def hard_prefilter(q_geom,cand,limits):
    n=len(np.asarray(cand['area'])); out=np.zeros(n,bool); wq,hq=q_geom.frame_w,q_geom.frame_h; T=limits.max_translate_frac*wq; ar=math.radians(limits.max_rot_deg); cq=np.array([wq/2,hq/2]); q=np.array([q_geom.cx,q_geom.cy])
    fw=np.asarray(cand['frame_w']); fh=np.asarray(cand['frame_h']); cx=np.asarray(cand['cx']); cy=np.asarray(cand['cy']); area=np.asarray(cand['area'])
    cf=np.maximum(wq/fw,hq/fh); pts=np.stack([cf*cx+(wq-cf*fw)/2, cf*cy+(hq-cf*fh)/2],axis=1); s=np.sqrt(q_geom.area/(area*cf*cf))
    valid=(s>=limits.min_scale)&(s<=limits.max_scale)
    # The rotated centroid traces a circular arc around the canvas centre.
    # Compute the exact closest point on that arc, rather than sampling
    # angles (sampling can create false negatives near the rotation bound).
    v=pts-cq; r=np.linalg.norm(v,axis=1)*s; target=q-cq; target_r=float(np.linalg.norm(target))
    theta=np.arctan2(v[:,1],v[:,0]); target_theta=math.atan2(float(target[1]),float(target[0]))
    delta=(target_theta-theta+math.pi)%(2*math.pi)-math.pi
    inside=np.abs(delta)<=ar
    radial=np.abs(r-target_r)
    def endpoint_dist(sign):
        a=theta+sign*ar
        return np.sqrt(np.maximum(0.0,r*r+target_r*target_r-2*r*target_r*np.cos(a-target_theta)))
    d=np.where(inside,radial,np.minimum(endpoint_dist(-1),endpoint_dist(1)))
    # A zero-radius centroid has no meaningful angle; the radial value is
    # already the exact distance in that case.
    valid &= d<=T+1e-9
    return valid
