from datasynth.normalizers import NormalizerChain

normalizer = NormalizerChain.execute(
    normalizer_template="examples/name_generator/normalizer.template",
    example="John and Jane Doe",
    temperature=0.0,
    cache=True,
    verbose=True,
    model_name="gpt-3.5-turbo",
)

print(normalizer)