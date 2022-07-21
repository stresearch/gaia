from gaia.export import export

# model_dir = "lightning_logs_transformer/version_10" #best model on old data
model_dir = "lightning_logs/version_1"
export_name = "export_model_cam4_v2.pt"

# inputs = ['FLNT', 'FLNS', 'T', 'SOLIN', 'Z3', 'OMEGA', 'FSNS', 'LHFLX', 'FSNT', 'SHFLX', 'U', 'V', 'Q', 'PSL']
# outputs = ['PRECC', 'PTTEND', 'PTEQ', 'Z3', 'PRECT']

inputs = None
# inputs = "Q,T,U,V,OMEGA,Z3,PS,SOLIN,SHFLX,LHFLX,FSNS,FLNS,FSNT,FLNT,FSDS".split(",")
outputs = "PTEQ,PTTEND,DCQ,DTCOND,QRS,QRL,CLOUD,CONCLD,FSNS,FLNS,FSNT,FLNT,FSDS,PRECT,PRECC,PRECL,PRECSC,PRECSL".split(",")


### v2
# Q,T,U,V,OMEGA,Z3,PS,SOLIN,SHFLX,LHFLX,FSNS,FLNS,FSNT,FLNT,FSDS
# CLOUD,CONCLD,DCQ,DTCOND,FLDS,PRECC,PRECL,PRECSC,PRECSL,PRECT,PSL,PTEQ,PTTEND,QRL,QRS,SOLL,SOLLD,SOLS,SOLSD,SRFRAD
# PTEQ,PTTEND,DCQ,DTCOND,QRS,QRL,CLOUD,CONCLD,FSNS,FLNS,FSNT,FLNT,FSDS,FLDS,SRFRAD,SOLL,SOLS,SOLLD,SOLSD,PSL,PRECT,PRECC,PRECL,PRECSC,PRECSL

export(model_dir, export_name, inputs=inputs, outputs=outputs)

