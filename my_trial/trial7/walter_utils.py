import jax
import jax.numpy as jnp
from mujoco import mjx


def get_sensor_data(model: mjx.Model, data: mjx.Data, sensor_name: str) -> jnp.ndarray:
    """Get sensor data from the model and data."""
    sensor_id = model.sensor(sensor_name).id
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    sensor_vector_adr = jnp.array(list(range(sensor_adr, sensor_adr + sensor_dim)))
    return data.sensordata[sensor_vector_adr].ravel()