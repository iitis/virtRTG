"""Ideal detector response for detector-plane X-ray signal arrays."""


class XRayDetector:
	"""Record incident signal with an ideal identity response.

	The detector is a computational component and owns no scene geometry. Its
	input is the simplified detector-plane signal currently produced by the
	projector, not calibrated photon fluence or detector counts.
	"""

	def record(self, incident_signal):
		"""Return the incident signal unchanged, without allocating an array copy."""
		return incident_signal
