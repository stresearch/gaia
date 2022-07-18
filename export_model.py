from gaia.export import export

model_dir = "lightning_logs_transformer/version_10"
export_name = "export_model_cam4_unnorm.pt"

# inputs = ['FLNT', 'FLNS', 'T', 'SOLIN', 'Z3', 'OMEGA', 'FSNS', 'LHFLX', 'FSNT', 'SHFLX', 'U', 'V', 'Q', 'PSL']
# outputs = ['PRECC', 'PTTEND', 'PTEQ', 'PRECT']

export(model_dir, export_name, inputs=None, outputs=None)

