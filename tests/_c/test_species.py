import mjp_inference as mjpi

protein = mjpi.Species(name='Protein', lower=0, upper=100, default=0)
print(protein.name, protein.lower, protein.upper, protein.default, protein.index)
protein.index = 8
print(protein.index)