from fuxion.normalizers import NormalizerChain

normalizer_chain = NormalizerChain.from_template(
    template_file="../templates/normalizer/address.template",
    temperature=0.0,
    cache=True,
    verbose=True,
    model_name="gpt-3.5-turbo",
)

normalizer = normalizer_chain.execute(
    example="John Doe street 1234, New York, NY 10001",
)
print(normalizer)