import numpy as np

def _normalize_vector(vector):
	"""Return a normalized 3D vector."""
	vector = np.asarray(vector, dtype=np.float32)
	norm = float(np.linalg.norm(vector))
	if norm <= 1e-8:
		raise ValueError("Vector norm must be greater than zero.")
	return vector / norm

def _transform_point(transform_matrix, point_xyz):
	"""Apply a 4x4 homogeneous transform to a 3D point."""
	point_h = np.ones(4, dtype=np.float32)
	point_h[:3] = np.asarray(point_xyz, dtype=np.float32)
	return (np.asarray(transform_matrix, dtype=np.float32) @ point_h)[:3]


def _transform_direction(transform_matrix, direction_xyz):
	"""Apply a 4x4 homogeneous transform to a 3D direction vector."""
	return np.asarray(transform_matrix, dtype=np.float32)[:3, :3] @ np.asarray(direction_xyz, dtype=np.float32)

