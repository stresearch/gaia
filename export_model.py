from gaia.export import export

model_dir = "lightning_logs_transformer/version_10"
export_name = "export_model_cam4_unnorm_test.pt"

inputs = ['FLNT', 'FLNS', 'T', 'SOLIN', 'Z3', 'OMEGA', 'FSNS', 'LHFLX', 'FSNT', 'SHFLX', 'U', 'V', 'Q', 'PSL']
outputs = ['PRECC', 'PTTEND', 'PTEQ', 'Z3', 'PRECT']

# outputs_og = "PTEQ,PTTEND,DCQ,DTCOND,QRS,QRL,CLOUD,CONCLD,FSNS,FLNS,FSNT,FLNT,FSDS,PRECT,PRECC,PRECL,PRECSC,PRECSL".split(",")

export(model_dir, export_name, inputs=inputs, outputs=outputs)

