import mjp_inference as mjpi

# make rate object
name = 'test'
value = 42.0
rate = mjpi.Rate(name=name, value=value)

print(rate)
print(rate.name)
print(rate.value)

