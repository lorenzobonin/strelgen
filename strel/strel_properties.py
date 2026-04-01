import torch
from strel.strel_advanced import Atom, Reach,  Globally, Eventually, Somewhere, Surround, Not, And, Or
from strel.strel_advanced import _compute_front_distance_matrix
import time
import math
import strel.strel_utils as su
import numpy as np
from enum import Enum
import utils.safety_metrics as saf


#####################################################
# THIS FILE CONTAINS EXAMPLES OF STREL PROPERTIES
# These properties are used in the paper, but they can be easily modified or new ones can be added.
# The properties defined have positive robustness when the scenario is unsafe, and negative robustness when the scenario is safe.
# softmax is used for numerical stability
# NEW PROPERTIES SHOULD BE DEFINED HERE
#####################################################

######################################################
# AGENT TYPES TO DEFINE NEW PROPERTIES
######################################################



# class syntax
class Agent(Enum):
    VEHICLE = 0
    PEDESTRIAN = 1
    CYCLIST = 2
    MOTORCYCLIST = 3
    BUS = 4
    STATIC = 5
    BACKGROUND = 6
    CONSTRUCTION = 7
    RIDERLESS_BICYCLE = 8
    UNKNOWN = 9





######################################################
# PROPERTIES
######################################################

def evaluate_reach_fast_slow(full_world, node_types, d_zone=20.0):
    traj = su.reshape_trajectories(full_world, node_types)
    N, T, _ = full_world.shape

    veh_labels = [0,3,4]



    slow_atom = And(Atom(4, threshold=0.01, lte=False), Atom(4, threshold=0.5, lte=True))

    fast_atom = And(Atom(4, threshold=2.2, lte=False), Atom(4, threshold=10, lte=True))


    reach = Reach(
        left_child=fast_atom,
        right_child=slow_atom,
        d1=0.01, d2=d_zone,
        distance_function="Front",
        left_label=veh_labels,
        right_label=veh_labels
    )

    
    prop = Eventually(reach, unbound=True)
    vals = prop.quantitative(traj, normalize=True).squeeze(2)[0]
    print(f"vals min: {vals.min()} max: {vals.max()}")
    return (1.0/20.0)*torch.logsumexp(20.0*vals.reshape(-1), dim=0)



def evaluate_ped_somewhere_unmask(full_world, node_types, d_zone=20.0):
    traj = su.reshape_trajectories(full_world, node_types)

    ped_labels = [1,2]
    veh_labels = [0,3,4]

    ped_atom = And(Atom(4, threshold=0.01, lte=False), Atom(4, threshold=0.3, lte=True))

    veh_atom = And(Atom(4, threshold=4.6, lte=False), Atom(4, threshold=6, lte=True))


    reach = Reach(
        left_child=veh_atom,
        right_child=ped_atom,
        d1=0.01, d2=d_zone,
        distance_function="Euclid",
        left_label=veh_labels,
        right_label=ped_labels
    )

    
    prop = Eventually(reach, unbound=True)
    vals = prop.quantitative(traj, normalize=True).squeeze(2)[0]
    print(f"vals min: {vals.min()} max: {vals.max()}")
    return (1.0/20.0)*torch.logsumexp(20.0*vals.reshape(-1), dim=0)




def evaluate_speeding_surrounded_unmask(
    full_world,
    node_types,
    v_fast=2.0,
    v_neigh_max=0.3,
    d_zone=3.0,   
):
    """
    Unsafe if: ∃ vehicle with high speed that is SURROUNDED by slow vehicles.
    Positive robustness ⇒ unsafe.
    """

    device = full_world.device



    traj = su.reshape_trajectories(full_world, node_types)

    veh_like = [0,3, 4]        # VEHICLE + MOTORBIKE + BUS ONLY

    fast_atom = And(Atom(4, threshold=3.2, lte=False), Atom(4, threshold=8.0, lte=True))
    slow_atom = And(Atom(4, threshold=0.1, lte=False), Atom(4, threshold=0.5, lte=True)) # neighbors slow

    surround_slow = Surround(
        left_child=fast_atom,
        right_child=slow_atom,
        d2=d_zone,
        distance_function="Euclid",
        left_labels=veh_like,
        right_labels=veh_like,
        all_labels=list(range(10)),      
    )

    prop = Eventually(surround_slow, unbound=True)

    vals = prop.quantitative(traj, normalize=True).squeeze(2)[0]   # [N,T]
    print(f"vals min: {vals.min()} max: {vals.max()}")
    alpha = 20.0
    robustness = (1.0 / alpha) * torch.logsumexp(alpha * vals.reshape(-1), dim=0)
    return robustness


def evaluate_slowing_surrounded_unmask(
    full_world,
    node_types,
    v_fast=2.0,
    v_neigh_max=0.3,
    d_sur=3.0,   
):
    """
    Unsafe if: ∃ vehicle with high speed that is SURROUNDED by slow vehicles.
    Positive robustness ⇒ unsafe.
    """

    device = full_world.device

    traj = su.reshape_trajectories(full_world, node_types)

    veh_like = [0,3, 4]        #

    fast_atom = And(Atom(4, threshold=2.0, lte=False), Atom(4, threshold=8.0, lte=True))
    slow_atom = And(Atom(4, threshold=0.1, lte=False), Atom(4, threshold=0.3, lte=True)) 

    surround_slow = Surround(
        left_child=slow_atom,
        right_child=fast_atom,
        d2=d_sur,
        distance_function="Euclid",
        left_labels=veh_like,
        right_labels=veh_like,
        all_labels=list(range(10)),      
    )

    prop = Eventually(surround_slow, unbound=True)

    vals = prop.quantitative(traj, normalize=True).squeeze(2)[0]   
    print(f"vals min: {vals.min()} max: {vals.max()}")
    alpha = 20.0
    robustness = (1.0 / alpha) * torch.logsumexp(alpha * vals.reshape(-1), dim=0)
    return robustness


