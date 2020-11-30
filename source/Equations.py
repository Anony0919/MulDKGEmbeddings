import torch
#import tensorflow as tf
MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}
################## EUCLIDEAN FUNCTIONS #################
def euc_sq_distance(x, y, eval_mode=False):
    """Computes Euclidean squared distance.

    Args:
      x: Tensor of size B1 x d
      y: Tensor of size B2 x d
      eval_mode: boolean indicating whether to compute all pairwise distances or
        not. If eval_mode=False, must have B1=B2.

    Returns:
      Tensor of size B1 x B2 if eval_mode=True, otherwise Tensor of size B1 x 1.
    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    if eval_mode:
        y2 = y2.t()
        xy = torch.matmul(x, y.t())
    else:
        xy = torch.sum(x * y, dim=-1, keepdim=True)
    return x2 + y2 - 2 * xy

def euc_reflection(r, x):
    """Applies 2x2 reflections.

    Args:
      r: Tensor of size B x d representing relfection parameters per example.
      x: Tensor of size B x d representing points to reflect.

    Returns:
      Tensor of size B x s representing reflection of x by r.
    """
    batch_size = r.shape[0]
    givens = r.view(batch_size, -1, 2)
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True)
    x = x.view(batch_size, -1, 2)
    x_ref = givens[:, :, 0:1] * torch.cat([x[:, :, 0:1], -x[:, :, 1:]], dim=-1) \
            + givens[:, :, 1:] * torch.cat((x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_ref.view(batch_size, -1)

def euc_rotations(r, x):
    """Applies 2x2 rotations.

    Args:
      r: Tensor of size B x d representing rotation parameters per example.
      x: Tensor of size B x d representing points to rotate.

    Returns:
      Tensor of size B x s representing rotation of x by r.
    """
    batch_size = r.shape[0]
    givens = r.view(batch_size, -1, 2)
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True)
    x = x.view(batch_size, -1, 2)
    #print(givens.shape, x.shape)
    # x_rot = givens[:, :, 0:1] * torch.cat([x[:, :, 0:1], x[:, :, :1]], dim=-1) \
    #         + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat(
        (-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view(batch_size, -1)

def euc_uniformul(r, x):
    batch_size = r.shape[0]
    givens = r.view(batch_size, -1, 2)
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True)
    x = x.view(batch_size, -1, 2)
    #print(givens.shape, x.shape)
    # x_rot = givens[:, :, 0:1] * torch.cat([x[:, :, 0:1], x[:, :, :1]], dim=-1) \
    #         + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    x_rot = givens[:, :, 0:1] * x
    return x_rot.view(batch_size, -1)

def euc_rotations_test(r,x):
    batch_size = x.shape[0]
    givens = r.view(batch_size, -1, 2, 2)
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True)
    givens = givens.view(batch_size, -1, 2)
    x = x.view(batch_size, -1, 2)
    x_rot = givens[:, :, 0:1] * torch.cat([x[:, :, 0:1], x[:, :, 1:2]], dim=-1) \
            + givens[:, :, 1:2] * torch.cat((x[:, :, 1:2], -x[:, :, 0:1]), dim=-1)
    # x_ref = givens[:, :, 2:3] * torch.cat([x[:, :, 2:3], x[:, :, 3:4]], dim=-1) \
    #         + givens[:, :, 3:4] * torch.cat((-x[:, :, 3:4], x[:, :, 2:3]), dim=-1)
    # x_ref = x[:,:,2:]
    # x_rot = torch.cat([x_rot, x_ref], dim = 1)
    return x_rot.view(batch_size, -1)

################## MATH FUNCTIONS #################

def clamp_min(x, min): # min > 0
    return min + torch.relu(x - min)

def clamp_max(x, max): # max > 0
    return max - torch.relu(max - x)

def clamp_maxmin(x, min, max):
    x = min + torch.relu(x - min)
    return max - torch.relu(max - x)

def artanh(x):
    eps = BALL_EPS[x.dtype]
    #x = torch.clamp(x, min=-1 + eps, max=1 - eps)
    x = clamp_min(x, min = -1 + eps)
    x = clamp_max(x, max = 1 - eps)
    return torch.log((1 + x) / (1 - x)) / 2 #torch.atanh

def tanh(x):
    MAX_TANH_ARG = 15.0
    x = torch.clamp(x, min=-MAX_TANH_ARG, max=MAX_TANH_ARG)
    # x = clamp_min(x, min=-MAX_TANH_ARG)
    # x = clamp_max(x, max=MAX_TANH_ARG)
    return torch.tanh(x)

################## HYP OPS ########################
def expmap0(u, c):
    """Hyperbolic exponential map at zero in the Poincare ball model.

    Args:
        u: Tensor of size B x dimension representing tangent vectors.
        c: Tensor of size 1 representing the absolute hyperbolic curvature.

    Returns:
        Tensor of shape B x dimension.
    """
    sqrt_c = torch.sqrt(c)
    u_norm = clamp_min(torch.norm(u, dim=-1, keepdim=True), min = MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)

def logmap0(y, c):
    """Hyperbolic logarithmic map at zero in the Poincare ball model.

    Args:
        y: Tensor of size B x dimension representing hyperbolic points.
        c: Tensor of size 1 representing the absolute hyperbolic curvature.

    Returns:
        Tensor of shape B x dimension.
    """
    sqrt_c = torch.sqrt(c)
    y_norm = clamp_min(torch.norm(y, dim=-1, keepdim=True), min = MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)

def project(x, c):
    """Projects points to the Poincare ball.

    Args:
        x: Tensor of size B x dimension.
        c: Tensor of size 1 representing the absolute hyperbolic curvature.
    #tf.clip_by_norm(t=x, clip_norm=(1. - eps) / torch.sqrt(c), axes=[1])
    #orch.renorm(x, p=2, dim=1, maxnorm=clip_norm.item())  error
    Returns:
        Tensor of shape B x dimension where each row is a point that lies within
        the Poincare ball.
    """
    # eps = BALL_EPS[x.dtype]
    # clip_norm = (1. - eps) / torch.sqrt(c)
    # l2 = torch.norm(x, p=2, dim=1, keepdim=True)
    # cliped_l2 = l2.clamp(min=clip_norm.item()) #不好 不传梯度？还是其他原因
    # cliped_l2 = clamp_min(l2, clip_norm) #不好
    # value = x * clip_norm / cliped_l2
    value = x
    return value

def mobius_add(x, y, c):
    """Element-wise Mobius addition.

    Args:
        x: Tensor of size B x dimension representing hyperbolic points.
        y: Tensor of size B x dimension representing hyperbolic points.
        c: Tensor of size 1 representing the absolute hyperbolic curvature.

    Returns:
        Tensor of shape B x dimension representing the element-wise Mobius addition
        of x and y.
    """
    cx2 = c * torch.sum(x * x, dim=-1, keepdim=True)
    cy2 = c * torch.sum(y * y, dim=-1, keepdim=True)
    cxy = c * torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * cxy + cy2) * x + (1 - cx2) * y
    denom = 1 + 2 * cxy + cx2 * cy2
    return project(num /clamp_min(denom, min=MIN_NORM), c)

################## HYP DISTANCE ###################
def hyp_distance(x, y, c, eval_mode=False):
  """Hyperbolic distance on the Poincare ball.

  Args:
    x: Tensor of size B1 x d
    y: Tensor of size B2 x d
    c: Tensor of size 1 representing the absolute hyperbolic curvature.
    eval_mode: boolean indicating whether to compute all pairwise distances or
      not. If eval_mode=False, must have B1=B2.

  Returns:
    Tensor of size B1 x B2 if eval_mode=True, otherwise Tensor of size B1 x 1.
  """
  sqrt_c = torch.sqrt(c)
  x2 = torch.sum(x * x, dim=-1, keepdim=True)
  if eval_mode:
      y2 = torch.sum(y * y, dim=-1, keepdim=True).t()
      xy = torch.matmul(x, y.t())
  else:
      y2 = torch.sum(y * y, dim=-1, keepdim=True)
      xy = torch.sum(x * y, dim=-1, keepdim=True)
  c1 = 1 - 2 * c * xy + c * y2
  c2 = 1 - c * x2
  num = torch.sqrt(c1 * c1 * x2 + c2 * c2 * y2 - (2 * c1 * c2) * xy)
  denom = 1 - 2 * c * xy + c * c * x2 * y2
  pairwise_norm = num / clamp_min(denom, min=MIN_NORM)
  dist = artanh(sqrt_c * pairwise_norm)
  return 2 * dist / sqrt_c
