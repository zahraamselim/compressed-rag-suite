"""Energy estimation utilities."""


def estimate_energy(latency_ms_per_token: float, tdp_watts: float) -> float:
    """
    Estimate energy consumption per token.
    
    Args:
        latency_ms_per_token: Latency in ms per token
        tdp_watts: Thermal Design Power in watts
        
    Returns:
        Energy consumption in millijoules per token
    """
    energy_mj = (latency_ms_per_token / 1000) * tdp_watts * 1000
    return energy_mj
