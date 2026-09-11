import routing
import math
import numpy as np
import compute_temp
   
TEMP_LIMIT = 80

def evaluate(self, env_idx):
    total_wirelength, avg_wirelength = self.get_wirelength(env_idx)
    temp = self.get_temp(env_idx)
    cost = avg_wirelength + math.pow(max(temp - TEMP_LIMIT, 0), 1.3) / (1 + math.exp(TEMP_LIMIT - temp))
    reward = -cost

    return {
        "reward": float(reward),
        "cost": float(cost),
        "total_wirelength": float(total_wirelength),
        "avg_wirelength": float(avg_wirelength),
        "temperature": float(temp),
    }


def _cost_function(self, env_idx):
    result = evaluate(self, env_idx)

    return env_idx, round(result["reward"], 4), round(result["avg_wirelength"], 4), round(result["temperature"], 4)

def get_wirelength(self, env_idx):
    total_wirelength, avg_wirelength = routing.solve_Cplex(self.vec_system[env_idx])
    return total_wirelength, avg_wirelength

    
def get_temp(self, env_idx):
    chiplets_width1 = 1.0e-3 * np.array(self.vec_system[env_idx].width)  # [mm] -> [m]
    chiplets_height1 = 1.0e-3 * np.array(self.vec_system[env_idx].height)  # [mm] -> [m]
    assert (0 < len(self.vec_system[env_idx].x))
    chiplets_x = 1.0e-3 * np.array(self.vec_system[env_idx].x)  # [mm] -> [m]
    assert (0 < len(self.vec_system[env_idx].y))
    assert (len(self.vec_system[env_idx].x) == len(self.vec_system[env_idx].y))
    chiplets_y = 1.0e-3 * np.array(self.vec_system[env_idx].y)  # [mm] -> [m]

    chiplets_left = np.array([])
    chiplets_bottom = np.array([])

    for i in range(0, self.vec_system[env_idx].chiplet_count, 1):
        chiplets_left = np.append(chiplets_left, chiplets_x[i] - 0.5 * chiplets_width1[i])
        chiplets_bottom = np.append(chiplets_bottom, chiplets_y[i] - 0.5 * chiplets_height1[i])

    # -- load rself and rmutu table model from file
    rself_list = []
    rmutu_list = []
    for i in range(0, self.vec_system[env_idx].chiplet_count, 1):
        chiplet_name = "Chiplet_" + str(i)
        rself = np.loadtxt(self.vec_path[env_idx] + chiplet_name + ".rself", delimiter='\t')
        rmutu = np.loadtxt(self.vec_path[env_idx] + chiplet_name + ".rmutu", delimiter='\t')
        rself_list.append(rself)
        rmutu_list.append(rmutu)

    # -- this is the real temperature computation which will be called from RL
    tmax = 0.0
    for i in range(0, self.vec_system[env_idx].chiplet_count, 1):
        t1 = compute_temp.compute_temp(i, self.vec_system[env_idx].chiplet_count, self.INTP_SIZE, chiplets_left, chiplets_bottom, chiplets_width1,
                                           chiplets_height1, self.vec_system[env_idx].power, rself_list[i], rmutu_list[i])
        tmax = t1 if t1 > tmax else tmax

    return tmax[0]
