#config file adapted from code produced by George Crochiere
import argparse

def add_flags_from_config(parser, config_dict):

    def OrNone(default):
        def func(x):
            if x.lower() == "none":
                return None
            elif default is None:
                return str(x)
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser
config_args = {
    'training_config': {
        'x_dim': (10, 'number of hexahedral cells in x-axis'),
        'y_dim': (10, 'number of hexahedral cells in y-axis'),
        'z_dim': (80, 'number of hexahedral cells in z-axis'),
        'h': (0.0002977976, 'height of the chip'),
        'num_steps': (100, 'number of time steps'),
        'tol': (1e-14, 'padding for float point comparisons'),
        'sampling_interval':(4.347826086956521e-6, 'interval between steps'), 
        'h_c':(2.40598e4, 'heat transfer coefficient'),  
        'k_0':(100, 'silicon thermal conductivity W/m·K'),
        'k_1':(1.2, 'oxide thermal conductivity W/m·K'),
        'rho_silicon':(2330, 'Density silicon (kg/m³)'),
        'rho_oxide':(2650, 'oxide density kg/m³'),
        'c_silicon':(751.1, 'silicon specific heat J/kg·K'),
        'c_oxide':(680.0, ' oxide specific heat J/kg·K'),
        'active_thickness':(0.0000557976, 'thickness of oxide layer'),
        'silicon_thickness':(0.0000557976, 'thickness of silicon layer'),
        'Ta':(0, 'ambient temperature'),
        'Pm':(75, 'power max used for mesh training and trace generation (Watts)'),
        'pg_change':(10, 'how many time steps before random power changes in training blocks'),
        'num_modes':(10, 'number of POD modes to use'),
        'file_path':("/home/hogebotl/data", "absolute file directory for all data created"),
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)