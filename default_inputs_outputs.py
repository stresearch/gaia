

def variable_spec_2209():

    inputs = """B_Q [t+1]
B_T [t+1]
B_U [t+1]
B_V [t+1]
B_OMEGA [t+1]
B_Z3 [t+1]
B_PS [t+1]
SOLIN [t+1]
B_SHFLX [t+1]
B_LHFLX [t+1]
LANDFRAC [t]
OCNFRAC [t]
ICEFRAC [t]
FSNS [t]
FLNS [t]
FSNT [t]
FLNT [t]
FSDS [t]""".split("\n")
    # B_TS [t+1]""".split("\n")


    outputs = """A_PTTEND [t+1]
A_PTEQ [t+1]
FSNS [t+1]
FLNS [t+1]
FSNT [t+1]
FLNT [t+1]
FSDS [t+1]
FLDS [t+1]
SRFRAD [t+1]
SOLL [t+1]
SOLS [t+1]
SOLLD [t+1]
SOLSD [t+1]
PRECT [t+1]
PRECC [t+1]
PRECL [t+1]
PRECSC [t+1]
PRECSL [t+1]""".split("\n")

    return inputs, outputs


def variable_spec_2210_V1():

    inputs = """B_Q [t+1]
B_T [t+1]
B_U [t+1]
B_V [t+1]
B_OMEGA [t+1]
B_Z3 [t+1]
B_PS [t+1]
SOLIN [t+1]
B_SHFLX [t+1]
B_LHFLX [t+1]
LANDFRAC [t]
OCNFRAC [t]
ICEFRAC [t]
B_TS [t+1]""".split("\n")


    outputs = """A_PTTEND [t+1]
A_PTEQ [t+1]
FSNS [t+1]
FLNS [t+1]
FSNT [t+1]
FLNT [t+1]
FSDS [t+1]
FLDS [t+1]
SRFRAD [t+1]
SOLL [t+1]
SOLS [t+1]
SOLLD [t+1]
SOLSD [t+1]
PRECT [t+1]
PRECC [t+1]
PRECL [t+1]
PRECSC [t+1]
PRECSL [t+1]""".split("\n")

    return inputs, outputs

def variable_spec_2210_V2():

    inputs = """B_Q [t+1]
B_T [t+1]
B_U [t+1]
B_V [t+1]
B_OMEGA [t+1]
B_PS [t+1]
SOLIN [t+1]
B_SHFLX [t+1]
B_LHFLX [t+1]
LANDFRAC [t]
OCNFRAC [t]
ICEFRAC [t]
B_TS [t+1]""".split("\n")


    outputs = """A_PTTEND [t+1]
A_PTEQ [t+1]
FSNS [t+1]
FLNS [t+1]
FSNT [t+1]
FLNT [t+1]
FSDS [t+1]
FLDS [t+1]
SRFRAD [t+1]
SOLL [t+1]
SOLS [t+1]
SOLLD [t+1]
SOLSD [t+1]
PRECT [t+1]
PRECC [t+1]
PRECL [t+1]
PRECSC [t+1]
PRECSL [t+1]""".split("\n")

    return inputs, outputs



def variable_spec_2210_V3():

    inputs = """B_RELHUM [t+1]
B_T [t+1]
B_U [t+1]
B_V [t+1]
B_OMEGA [t+1]
B_Z3 [t+1]
B_PS [t+1]
SOLIN [t+1]
B_SHFLX [t+1]
B_LHFLX [t+1]
LANDFRAC [t]
OCNFRAC [t]
ICEFRAC [t]
B_TS [t+1]""".split("\n")


    outputs = """A_PTTEND [t+1]
A_PTEQ [t+1]
FSNS [t+1]
FLNS [t+1]
FSNT [t+1]
FLNT [t+1]
FSDS [t+1]
FLDS [t+1]
SRFRAD [t+1]
SOLL [t+1]
SOLS [t+1]
SOLLD [t+1]
SOLSD [t+1]
PRECT [t+1]
PRECC [t+1]
PRECL [t+1]
PRECSC [t+1]
PRECSL [t+1]""".split("\n")

    return inputs, outputs


def variable_spec_2210_V4():

    inputs = """B_RELHUM [t+1]
B_Q [t+1]
B_T [t+1]
B_U [t+1]
B_V [t+1]
B_OMEGA [t+1]
B_Z3 [t+1]
B_PS [t+1]
SOLIN [t+1]
B_SHFLX [t+1]
B_LHFLX [t+1]
LANDFRAC [t]
OCNFRAC [t]
ICEFRAC [t]
B_TS [t+1]""".split("\n")


    outputs = """A_PTTEND [t+1]
A_PTEQ [t+1]
FSNS [t+1]
FLNS [t+1]
FSNT [t+1]
FLNT [t+1]
FSDS [t+1]
FLDS [t+1]
SRFRAD [t+1]
SOLL [t+1]
SOLS [t+1]
SOLLD [t+1]
SOLSD [t+1]
PRECT [t+1]
PRECC [t+1]
PRECL [t+1]
PRECSC [t+1]
PRECSL [t+1]""".split("\n")

    return inputs, outputs



def variable_spec_2304_V5():

    inputs = """B_Q [t+1]
B_T [t+1]
B_U [t+1]
B_V [t+1]
B_OMEGA [t+1]
B_Z3 [t+1]
B_CLDLIQ [t+1]
B_CLDICE [t+1]
B_PS [t+1]
SOLIN [t+1]
B_SHFLX [t+1]
B_LHFLX [t+1]
LANDFRAC [t]
OCNFRAC [t]
ICEFRAC [t]
B_TS [t+1]""".split("\n")


    outputs = """A_PTTEND [t+1]
A_PTEQ [t+1]
A_PTECLDLIQ [t+1]
A_PTECLDICE [t+1]
FSNS [t+1]
FLNS [t+1]
FSNT [t+1]
FLNT [t+1]
FSDS [t+1]
FLDS [t+1]
SRFRAD [t+1]
SOLL [t+1]
SOLS [t+1]
SOLLD [t+1]
SOLSD [t+1]
PRECT [t+1]
PRECC [t+1]
PRECL [t+1]
PRECSC [t+1]
PRECSL [t+1]""".split("\n")

    return inputs, outputs


def variable_spec_2210_V6():

    inputs = """B_Q [t+1]
B_T [t+1]
B_U [t+1]
B_V [t+1]
B_OMEGA [t+1]
B_Z3 [t+1]
B_PS [t+1]
SOLIN [t+1]
B_SHFLX [t+1]
B_LHFLX [t+1]
LANDFRAC [t]
OCNFRAC [t]
ICEFRAC [t]
B_TS [t+1]
PRECT [t]
PRECC [t]
PRECL [t]
PRECSC [t]
PRECSL [t]""".split("\n")


    outputs = """A_PTTEND [t+1]
A_PTEQ [t+1]
FSNS [t+1]
FLNS [t+1]
FSNT [t+1]
FLNT [t+1]
FSDS [t+1]
FLDS [t+1]
SRFRAD [t+1]
SOLL [t+1]
SOLS [t+1]
SOLLD [t+1]
SOLSD [t+1]
PRECT [t+1]
PRECC [t+1]
PRECL [t+1]
PRECSC [t+1]
PRECSL [t+1]""".split("\n")

    return inputs, outputs


