import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import torch

def add_hyper_rectangle(ax, hyper_rectangle_predictor, x_specific, color='blue', linestyle='-'):
    """
    Adds a rectangle to the given plot `ax` based on predictions calculated
    from the `hyper_rectangle_predictor` and `x_specific`.

    Parameters:
        ax: matplotlib.axes.Axes
            The plot where the rectangle will be added.
        hyper_rectangle_predictor: Object
            Contains the prediction models and conformal value for bounds.
        x_specific: numpy.ndarray or similar
            The specific input for which predictions are calculated.
    """
    # Calculate predictions
    k = 2  # Assuming 2 dimensions for this example
    predictions = np.zeros((k, 2), dtype=float)

    for i in range(k):
        predictions[i, 0] = hyper_rectangle_predictor.tab_model_alpha_low[i](
            torch.tensor(x_specific, dtype=torch.float32).reshape(1, -1)
        ).item() - hyper_rectangle_predictor.conformal_value

        predictions[i, 1] = hyper_rectangle_predictor.tab_model_alpha_high[i](
            torch.tensor(x_specific, dtype=torch.float32).reshape(1, -1)
        ).item() + hyper_rectangle_predictor.conformal_value

    # Extract bounds for rectangle
    x_min, x_max = predictions[0]
    y_min, y_max = predictions[1]

    # Compute the width and height of the rectangle
    width = x_max - x_min
    height = y_max - y_min

    # Add the rectangle to the plot
    rectangle = Rectangle(
        (x_min, y_min), width, height,
        edgecolor='blue', facecolor='none', linestyle=linestyle, linewidth=2
    )
    ax.add_patch(rectangle)
    return ax


def add_ellipse(ax, center, Lambda, q, conformal_value, color='red', linestyle = '-', label='Conformal Ellipse'):
    """
       plot -> {y, \| \Lambda (u - center) \|_q^q \leq conformal_value}  }
       conformal_value = radius ** q
    """

    # Generate a set of points on a unit circle
    theta = np.linspace(0, 2 * np.pi, 500)
    unit_circle = np.vstack((np.cos(theta), np.sin(theta)))
    
    # Scale the unit circle points to match the conformal value under the q-norm
    scaling = conformal_value * np.power(np.sum(np.abs(unit_circle) ** q, axis=0), -1/q)
    scaled_points = unit_circle * scaling
    
    # Apply the linear transformation Lambda
    transformed_points = np.linalg.inv(Lambda) @ scaled_points
    
    # Translate the ellipse to the center
    ellipse_points = transformed_points + np.reshape(center, (2, 1))
    
    # Add the ellipse to the existing plot
    ax.plot(ellipse_points[0, :], ellipse_points[1, :], label=label, c=color, linestyle=linestyle)
   #  ax.scatter(*center, color='red', label='Center')
    ax.legend()
    
    return ax



def old_add_ellipse_multiple_lambdas(ax, f_x, rotation, tab_diag, tab_q, tab_weights, conformal_value, color='red', label='Conformal Ellipse'):
   """
      plot -> {y, \| \Lambda (u - center) \|_q^q \leq conformal_value}  }
      conformal_value = radius ** q
   """
   cmap = cm.get_cmap('tab10', len(tab_diag))
   cumulative_weight = np.cumsum(tab_weights)
   for i in range(len(tab_diag)):
      Lambda =  tab_diag[i]
      # Generate a set of points on a unit circle
      # theta = np.linspace(0, 2 * np.pi, 500)
      
      if i > 0:
         theta = np.linspace(2 * np.pi * cumulative_weight[i-1], 2 * np.pi * cumulative_weight[i], 500)
      else:
         theta = np.linspace(0, 2 * np.pi * cumulative_weight[i], 500)
      
      unit_circle = np.vstack((np.cos(theta), np.sin(theta)))
      
      # Scale the unit circle points to match the conformal value under the q-norm
      scaling = conformal_value * np.power(np.sum(np.abs(unit_circle) ** tab_q[i], axis=0), -1/tab_q[i])
      scaled_points = unit_circle * scaling
      
      # Apply the linear transformation Lambda
      transformed_points = np.linalg.inv(Lambda) @ scaled_points
      
      # Translate the ellipse to the center
      ellipse_points = rotation.T @ transformed_points + f_x[:, None]
      
      # Add the ellipse to the existing plot
      ax.plot(ellipse_points[0, :], ellipse_points[1, :], c=cmap(i))
      if i != 0:
         first_point = ellipse_points[:, 0]
         ax.plot([last_point[0], first_point[0]], [last_point[1], first_point[1]], c=color)
 
      else:
         first_point_boucle = ellipse_points[:, 0].copy()
      
      last_point = ellipse_points[:, -1]
      
      if i == len(tab_diag) - 1:
         ax.plot([last_point[0], first_point_boucle[0]], [last_point[1], first_point_boucle[1]], c="red")
      
      ax.legend()
   
   return ax


def add_ellipse_multiple_lambdas(ax, f_x, rotation, tab_diag, tab_q, tab_weights, split, conformal_value, color='red', label='Conformal Ellipse'):
   """
      plot -> {y, \| \Lambda (u - center) \|_q^q \leq conformal_value}  }
      conformal_value = radius ** q
   """
   cmap = cm.get_cmap('tab10', len(tab_diag))
   tab_weights = np.array([1/4, 1/4, 1/4, 1/4])
   cumulative_weight = np.cumsum(tab_weights)
   points = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float32)
   idx_lambdas = split.get_idx(points)
   for j in range(4):
      i = idx_lambdas[j]
      Lambda =  tab_diag[i]
      # Generate a set of points on a unit circle      
      if j > 0:
         theta = np.linspace(2 * np.pi * cumulative_weight[j-1], 2 * np.pi * cumulative_weight[j], 500)
      else:
         theta = np.linspace(0, 2 * np.pi * cumulative_weight[j], 500)
      
      unit_circle = np.vstack((np.cos(theta), np.sin(theta)))
      
      # Scale the unit circle points to match the conformal value under the q-norm
      scaling = conformal_value * np.power(np.sum(np.abs(unit_circle) ** tab_q[i], axis=0), -1/tab_q[i])
      scaled_points = unit_circle * scaling
      
      # Apply the linear transformation Lambda
      transformed_points = np.linalg.inv(Lambda) @ scaled_points
      
      # Translate the ellipse to the center
      ellipse_points = rotation.T @ transformed_points + f_x[:, None]
      
      # Add the ellipse to the existing plot
      ax.plot(ellipse_points[0, :], ellipse_points[1, :], c=cmap(i))
      if j != 0:
         first_point = ellipse_points[:, 0]
         ax.plot([last_point[0], first_point[0]], [last_point[1], first_point[1]], c=color)
 
      else:
         first_point_boucle = ellipse_points[:, 0].copy()
      
      last_point = ellipse_points[:, -1]
      
      if j == len(tab_diag) - 1:
         ax.plot([last_point[0], first_point_boucle[0]], [last_point[1], first_point_boucle[1]], c="red")
      
      ax.legend()
   
   return ax


def surface_ellipse_3d(center, Lambda, radius):
    """
       plot -> {y, \| \Lambda (u - center) \|_q^q \leq conformal_value}  }
       conformal_value = radius ** q
    """
    q = 2
    # Generate points on a unit sphere
    u = np.linspace(0, 2 * np.pi, 200)
    v = np.linspace(0, np.pi, 200)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    unit_sphere = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    
    # Scale points to match the conformal value under the q-norm
   #  scaling = radius ** (1/q) * np.power(np.sum(np.abs(unit_sphere) ** q, axis=0), -1/q)
    scaling = radius * np.power(np.sum(np.abs(unit_sphere) ** q, axis=0), -1/q)
    scaled_points = unit_sphere * scaling
    
    # Apply the linear transformation Lambda
    transformed_points = np.linalg.inv(Lambda) @ scaled_points
    
    # Translate the ellipsoid to the center
    ellipsoid_points = transformed_points + np.reshape(center, (3, 1))
    
    # Reshape to original grid shape
    X = ellipsoid_points[0, :].reshape(x.shape)
    Y = ellipsoid_points[1, :].reshape(y.shape)
    Z = ellipsoid_points[2, :].reshape(z.shape)
    
    return [X, Y, Z]






########## CLASSIFICATION PLOT ##########


def softmax(x, axis=0):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def surface_ellipse_3d_simplex(center, Lambda, radius):
    """
       plot -> {y, \| \Lambda (u - center) \|_q^q \leq conformal_value}  }
       conformal_value = radius ** q
    """
    q = 2
    # Generate points on a unit sphere
    u = np.linspace(0, 2 * np.pi, 200)
    v = np.linspace(0, np.pi, 200)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    unit_sphere = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    
    # Scale points to match the conformal value under the q-norm
   #  scaling = radius ** (1/q) * np.power(np.sum(np.abs(unit_sphere) ** q, axis=0), -1/q)
    scaling = radius * np.power(np.sum(np.abs(unit_sphere) ** q, axis=0), -1/q)
    scaled_points = unit_sphere * scaling
    
    # Apply the linear transformation Lambda
    transformed_points = np.linalg.inv(Lambda) @ scaled_points
    
    # Translate the ellipsoid to the center
    ellipsoid_points = transformed_points + np.reshape(center, (3, 1))
    ellipsoid_points = softmax(ellipsoid_points)
    
    # Reshape to original grid shape
    X = ellipsoid_points[0, :].reshape(x.shape)
    Y = ellipsoid_points[1, :].reshape(y.shape)
    Z = ellipsoid_points[2, :].reshape(z.shape)
    
    return [X, Y, Z]


def surface_ellipse_simplex(center, Lambda, radius):
    """
       plot -> {y, \| \Lambda (u - center) \|_q^q \leq conformal_value}  }
       conformal_value = radius ** q
    """
    q = 2
    # Generate points on a unit sphere
    u = np.linspace(0, 2 * np.pi, 200)
    v = np.linspace(0, np.pi, 200)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    unit_sphere = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    
    # Scale points to match the conformal value under the q-norm
   #  scaling = radius ** (1/q) * np.power(np.sum(np.abs(unit_sphere) ** q, axis=0), -1/q)
    scaling = radius * np.power(np.sum(np.abs(unit_sphere) ** q, axis=0), -1/q)
    scaled_points = unit_sphere * scaling
    
    # Apply the linear transformation Lambda
    transformed_points = np.linalg.inv(Lambda) @ scaled_points
    
    # Translate the ellipsoid to the center
    ellipsoid_points = transformed_points + np.reshape(center, (3, 1))
    
    ellipsoid_points = softmax(ellipsoid_points)
    
    print("e", ellipsoid_points.shape)
    
    ellipse_points = simplex3_to_2D(ellipsoid_points.T)
    print('e2', ellipse_points.shape)
    return ellipse_points


def simplex3_to_2D(points):
    """
    Projette des points du simplexe 3D (somme = 1, coordonnées ≥ 0) en 2D pour affichage dans un triangle.
    
    Args:
        points (ndarray): tableau de forme (n, 3), chaque ligne est un point (x, y, z) dans le simplexe.

    Returns:
        ndarray: tableau de forme (n, 2), les coordonnées 2D projetées.
    """
    # Sommets du triangle équilatéral en 2D
    v1 = np.array([0, 0])
    v2 = np.array([1, 0])
    v3 = np.array([0.5, np.sqrt(3)/2])

    # On fait une combinaison convexe des sommets
    return points[:, 0][:, None] * v1 + points[:, 1][:, None] * v2 + points[:, 2][:, None] * v3


