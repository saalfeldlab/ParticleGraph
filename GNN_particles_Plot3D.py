import matplotlib.cm as cmplt
from matplotlib.ticker import FormatStrFormatter
from torch_geometric.nn import MessagePassing
import torch_geometric.utils as pyg_utils
import os
from ParticleGraph.MLP import MLP
import imageio


from ParticleGraph.generators import RD_RPS
from ParticleGraph.models import Interaction_Particles, Mesh_Laplacian
from ParticleGraph.fitting_models import power_model, boids_model, reaction_diffusion_model

os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

# from data_loaders import *
from GNN_particles_Ntype import *
from ParticleGraph.embedding_cluster import *
from ParticleGraph.utils import to_numpy, CustomColorMap, choose_boundary_values

# matplotlib.use("Qt5Agg")

import jax
import jax.numpy as jp

def norm(v, axis=-1, keepdims=False, eps=0.0):
  return jp.sqrt((v*v).sum(axis, keepdims=keepdims).clip(eps))

def normalize(v, axis=-1, eps=1e-20):
  return v/norm(v, axis, keepdims=True, eps=eps)

import io
import base64
import time
from functools import partial
from typing import NamedTuple
import subprocess

import PIL
import numpy as np
import matplotlib.pylab as pl

from IPython.display import display, Image, HTML

def np2pil(a):
  if a.dtype in [np.float32, np.float64]:
    a = np.uint8(np.clip(a, 0, 1) * 255)
  return PIL.Image.fromarray(a)


def imwrite(f, a, fmt=None):
  a = np.asarray(a)
  if isinstance(f, str):
    fmt = f.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg':
      fmt = 'jpeg'
    f = open(f, 'wb')
  np2pil(a).save(f, fmt, quality=95)


def imencode(a, fmt='jpeg'):
  a = np.asarray(a)
  if len(a.shape) == 3 and a.shape[-1] == 4:
    fmt = 'png'
  f = io.BytesIO()
  imwrite(f, a, fmt)
  return f.getvalue()


def imshow(a, fmt='jpeg', display=display):
  return display(Image(data=imencode(a, fmt)))


class VideoWriter:
  def __init__(self, filename='_autoplay.mp4', fps=30.0):
    self.ffmpeg = None
    self.filename = filename
    self.fps = fps
    self.view = display(display_id=True)
    self.last_preview_time = 0.0

  def add(self, img):
    img = np.asarray(img)
    h, w = img.shape[:2]
    if self.ffmpeg is None:
      self.ffmpeg = self._open(w, h)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1) * 255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.ffmpeg.stdin.write(img.tobytes())
    t = time.time()
    if self.view and t - self.last_preview_time > 1:
      self.last_preview_time = t
      imshow(img, display=self.view.update)

  def __call__(self, img):
    return self.add(img)

  def _open(self, w, h):
    cmd = f'''ffmpeg -y -f rawvideo -vcodec rawvideo -s {w}x{h}
      -pix_fmt rgb24 -r {self.fps} -i - -pix_fmt yuv420p 
      -c:v libx264 -crf 20 {self.filename}'''.split()
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

  def close(self):
    if self.ffmpeg:
      self.ffmpeg.stdin.close()
      self.ffmpeg.wait()
      self.ffmpeg = None

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()
    if self.filename == '_autoplay.mp4':
      self.show()

  def show(self):
    self.close()
    if not self.view:
      return
    b64 = base64.b64encode(open(self.filename, 'rb').read()).decode('utf8')
    s = f'''<video controls loop>
 <source src="data:video/mp4;base64,{b64}" type="video/mp4">
 Your browser does not support the video tag.</video>'''
    self.view.update(HTML(s))


def animate(f, duration_sec, fps=60):
  with VideoWriter(fps=fps) as vid:
    for t in jp.linspace(0, 1, int(duration_sec * fps)):
      vid(f(t))




class Balls(NamedTuple):
  pos: jp.ndarray
  color: jp.ndarray

def create_balls(key, n=16, R=3.0):
  pos, color = jax.random.uniform(key, [2, n, 3])
  pos = (pos-0.5)*R
  return Balls(pos, color)



def balls_sdf(balls, p, ball_r=0.5):
  dists = norm(p-balls.pos)-ball_r
  return dists.min()

if __name__ == '__main__':

  key = jax.random.PRNGKey(123)
  balls = create_balls(key)

  p = jax.random.normal(key, [1000, 3])
  print( jax.vmap(partial(balls_sdf, balls))(p).shape )

  show_slice(partial(balls_sdf, balls), z=0.0);











