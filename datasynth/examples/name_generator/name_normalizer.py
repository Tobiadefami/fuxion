from datasynth.normalizers import NormalizerChain

normalizer_chain = NormalizerChain.from_template(
    template_file="examples/name_generator/normalizer.template",
    temperature=0.0,
    cache=True,
    verbose=True,
    model_name="gpt-3.5-turbo",
)

normalizer = normalizer_chain.execute(
    example="John Doe",
)
print(normalizer)