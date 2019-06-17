from ges_synthetic_data.synthesizeGesData3 import SynthesizeData

data_factory = SynthesizeData('ges-health-problems.json')
data_factory.create_synthesize_data('new_data_all_ages.csv')
data_factory.create_synthesize_data('only_random.csv', example_for_each_age=False)
