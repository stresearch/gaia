from gaia.export import export

model_dir = "lightning_logs_hparam/version_0"
export_name = "export_model_cam4.pt"

inputs = ['FLNT', 'FLNS', 'T', 'SOLIN', 'Z3', 'OMEGA', 'FSNS', 'LHFLX', 'FSNT', 'SHFLX', 'U', 'V', 'Q', 'PSL']
outputs = ['PRECC', 'PTTEND', 'PTEQ', 'PRECT']

export(model_dir,export_name, inputs, outputs)

