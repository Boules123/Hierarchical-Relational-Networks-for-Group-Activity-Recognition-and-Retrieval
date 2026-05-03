from src.models.non_temporal.B1_NoRelations import B1_NoRelations
from src.models.non_temporal.B2_RCRG_1R_1C import B2_RCRG_1R_1C
from src.models.non_temporal.B3_RCRG_1R_1C_notTuned import B3_RCRG_1R_1C_notTuned
from src.models.non_temporal.B4_RCRG_2R_11C import B4_RCRG_2R_11C
from src.models.non_temporal.B4_RCRG_2R_11C_conc import B4_RCRG_2R_11C_conc
from src.models.non_temporal.B5_RCRG_2R_21C import B5_RCRG_2R_21C
from src.models.non_temporal.B5_RCRG_2R_21C_conc import B5_RCRG_2R_21C_conc
from src.models.non_temporal.B6_RCRG_3R_421C import B6_RCRG_3R_421C
from src.models.non_temporal.B6_RCRG_3R_421C_conc import B6_RCRG_3R_421C_conc


from src.models.temporal.RCRG_2R_21C import RCRG_2R_21C 
from src.models.temporal.RCRG_2R_11C_conc import RCRG_2R_21C_conc

model_registery = {
    "B1_NoRelations": B1_NoRelations,
    "B2_RCRG_1R_1C": B2_RCRG_1R_1C,
    "B3_RCRG_1R_1C_notTuned": B3_RCRG_1R_1C_notTuned,
    "B4_RCRG_2R_11C": B4_RCRG_2R_11C,
    "B4_RCRG_2R_11C_conc": B4_RCRG_2R_11C_conc,
    "B5_RCRG_2R_21C": B5_RCRG_2R_21C,
    "B5_RCRG_2R_21C_conc": B5_RCRG_2R_21C_conc,
    "B6_RCRG_3R_421C": B6_RCRG_3R_421C,
    "B6_RCRG_3R_421C_conc": B6_RCRG_3R_421C_conc,
    
    # temporal 
    "RCRG_2R_21C": RCRG_2R_21C,
    "RCRG_2R_11C_conc": RCRG_2R_21C_conc
}

def get_model(model_name, person_cls):
    return model_registery[model_name](person_cls=person_cls)
