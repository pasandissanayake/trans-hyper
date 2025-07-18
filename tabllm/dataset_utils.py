import random
from scipy import spatial
from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


# from .helper.note_generator import NoteGenerator
# from .helper.note_template import NoteTemplate
from .helper.external_datasets_variables import *
# from .helper.preprocess import preprocess


def load_dataset(dataset_name, data_dir):
    def byte_to_string_columns(data):
        for col, dtype in data.dtypes.items():
            if dtype == object:  # Only process byte object columns.
                data[col] = data[col].apply(lambda x: x.decode("utf-8"))
        return data

    if dataset_name == "creditg":
        dataset = pd.DataFrame(loadarff(data_dir / 'dataset_31_credit-g.arff')[0])
        dataset = byte_to_string_columns(dataset)
        dataset.rename(columns={'class': 'label'}, inplace=True)
        dataset['label'] = dataset['label'] == 'good'

    elif dataset_name == "blood":
        columns = {'V1': 'recency', 'V2': 'frequency', 'V3': 'monetary', 'V4': 'time', 'Class': 'label'}
        dataset = pd.DataFrame(loadarff(data_dir / 'php0iVrYT.arff')[0])
        dataset = byte_to_string_columns(dataset)
        dataset.rename(columns=columns, inplace=True)
        dataset['label'] = dataset['label'] == '2'
        
    elif dataset_name == "bank":
        columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
                   'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
        columns = {'V' + str(i + 1): v for i, v in enumerate(columns)}
        dataset = pd.DataFrame(loadarff(data_dir / 'phpkIxskf.arff')[0])
        dataset = byte_to_string_columns(dataset)
        dataset.rename(columns=columns, inplace=True)
        dataset.rename(columns={'Class': 'label'}, inplace=True)
        dataset['label'] = dataset['label'] == '2'
        
    elif dataset_name == "jungle":
        dataset = pd.DataFrame(loadarff(data_dir / 'jungle_chess_2pcs_raw_endgame_complete.arff')[0])
        dataset = byte_to_string_columns(dataset)
        dataset.rename(columns={'class': 'label'}, inplace=True)
        dataset['label'] = dataset['label'] == 'w'  # Does white win?
        
    elif dataset_name == "calhousing":
        dataset = pd.DataFrame(loadarff(data_dir / 'houses.arff')[0])
        dataset = byte_to_string_columns(dataset)
        dataset.rename(columns={'median_house_value': 'label'}, inplace=True)
        # Make binary task by labelling upper half as true
        median_price = dataset['label'].median()
        dataset['label'] = dataset['label'] > median_price
        
    elif dataset_name == "income":
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                   'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                   'native_country', 'label']

        def strip_string_columns(df):
            df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.str.strip())

        dataset_train = pd.read_csv(data_dir / 'adult.data', names=columns, na_values=['?', ' ?'])
        dataset_train = dataset_train.drop(columns=['fnlwgt', 'education_num'])
        original_size = len(dataset_train)
        strip_string_columns(dataset_train)
        # Multiply all dollar columns by two to adjust for inflation
        # dataset_train[['capital_gain', 'capital_loss']] = (1.79 * dataset_train[['capital_gain', 'capital_loss']]).astype(int)
        dataset_train['label'] = dataset_train['label'] == '>50K'

        dataset_test = pd.read_csv(data_dir / 'adult.test', names=columns, na_values=['?', ' ?'])
        dataset_test = dataset_test.drop(columns=['fnlwgt', 'education_num'])
        strip_string_columns(dataset_test)
        # Note label string in test set contains full stop
        # dataset_test[['capital_gain', 'capital_loss']] = (1.79 * dataset_test[['capital_gain', 'capital_loss']]).astype(int)
        dataset_test['label'] = dataset_test['label'] == '>50K.'

        dataset = pd.concat([dataset_train, dataset_test])
        # dataset_train, dataset_valid = train_test_split(dataset_train, test_size=0.20, random_state=1)
        # dataset = dataset_train
        # assert len(dataset_train) + len(dataset_valid) == original_size

    elif dataset_name == "car":
        columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety_dict', 'label']
        dataset = pd.read_csv(data_dir / 'car.data', names=columns)
        original_size = len(dataset)
        label_dict = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
        dataset['label'] = dataset['label'].replace(label_dict)
        
    elif dataset_name == "voting":
        columns = ['label', 'handicapped_infants', 'water_project_cost_sharing', 'adoption_of_the_budget_resolution',
                   'physician_fee_freeze', 'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban',
                   'aid_to_nicaraguan_contras', 'mx_missile', 'immigration', 'synfuels_corporation_cutback',
                   'education_spending', 'superfund_right_to_sue', 'crime', 'duty_free_exports',
                   'export_administration_act_south_africa']
        dataset = pd.read_csv(data_dir / 'house-votes-84.data', names=columns, na_values=['?'])
        original_size = len(dataset)
        dataset['label'] = np.where(dataset['label'] == 'democrat', 1, 0)
        
    elif dataset_name == "wine":
        columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
                   'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
        dataset = pd.read_csv(data_dir / 'winequality-red.csv', names=columns, skiprows=[0])
        original_size = len(dataset)
        # Adopt grouping from: https://www.kaggle.com/code/vishalyo990/prediction-of-quality-of-wine
        bins = (2, 6.5, 8)
        dataset['quality'] = pd.cut(dataset['quality'], bins=bins, labels=[0, 1]).astype(int)  # bad, good
        dataset = dataset.rename(columns={'quality': 'label'})
        
    elif dataset_name == "titanic":
        # Only use training set since no labels for test set
        dataset = pd.read_csv(data_dir / 'train.csv')
        original_size = len(dataset)
        dataset = dataset.rename(columns={'Survived': 'label'})
        
    elif dataset_name == "heart":
        dataset = pd.read_csv(data_dir / 'heart.csv')
        original_size = len(dataset)
        dataset = dataset.rename(columns={'HeartDisease': 'label'})
        
    elif dataset_name == "diabetes":
        dataset = pd.read_csv(data_dir / 'diabetes.csv')
        original_size = len(dataset)
        dataset = dataset.rename(columns={'Outcome': 'label'})
        
    else:
        raise ValueError("Dataset not found")

    # For final experiments, ensure correct numbers of features for each dataset
    dataset_specs = {
        'income': 13,
        'car': 7,
        'heart': 12,
        'diabetes': 9,
        'creditg': 21,
        'blood': 5,
        'bank': 17,
        'jungle': 7,
        'wine': 12,
        'calhousing': 9
    }
    assert dataset_name in dataset_specs.keys() and len(dataset.columns) == dataset_specs[dataset_name]
    return dataset


def load_and_preprocess_dataset(dataset_name, data_dir):
    dataset = load_dataset(dataset_name=dataset_name, data_dir=data_dir)

    if dataset_name == "bank":
        categorical = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
        numerical = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    elif dataset_name == "blood":
        categorical = []
        numerical = ['recency', 'frequency', 'monetary', 'time']

    elif dataset_name == "calhousing":
        categorical = []
        numerical = ['latitude', 'population', 'median_income', 'longitude', 'total_bedrooms', 'housing_median_age', 'households', 'total_rooms']

    elif dataset_name == "car":
        categorical = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety_dict']
        numerical = []
    
    elif dataset_name == "creditg":
        categorical = ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment', 'personal_status', 'other_parties', 'residence_since', 'property_magnitude', 'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker']
        numerical = ['num_dependents', 'existing_credits', 'installment_commitment', 'credit_amount', 'duration', 'age']

    elif dataset_name == "diabetes":
        categorical = []
        numerical = ['Age', 'Insulin', 'BloodPressure', 'Glucose', 'DiabetesPedigreeFunction', 'BMI', 'SkinThickness', 'Pregnancies']
        
    elif dataset_name == "heart":
        categorical = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        numerical = ['Oldpeak', 'Age', 'FastingBS', 'MaxHR', 'Cholesterol', 'RestingBP']

    elif dataset_name == "income":
        categorical = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
        numerical = ['hours_per_week', 'capital_gain', 'capital_loss', 'age']
        
    elif dataset_name == "jungle":
        categorical = []
        numerical = ['white_piece0_rank', 'white_piece0_file', 'black_piece0_strength', 'black_piece0_rank', 'white_piece0_strength', 'black_piece0_file']
            
    elif dataset_name == "voting":
        categorical = []
        numerical = []
        
    elif dataset_name == "wine":
        categorical = []
        numerical = []
        
    elif dataset_name == "titanic":
        categorical = []
        numerical = []
               
    else:
        raise ValueError("Dataset not found")

    dataset = preprocess_numcols(dataset, numerical)
    dataset = preprocess_catcols(dataset, categorical)
    return dataset


def preprocess_numcols(df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    """
    Normalize and center numerical features (zero mean, unit variance).

    Args:
        df (pd.DataFrame): Input dataframe.
        numerical_cols (list): List of column names to normalize.

    Returns:
        pd.DataFrame: A new dataframe with specified columns normalized.
    """
    if not numerical_cols:  # safely skip if list is empty
        return df
    
    df_processed = df.copy()
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    return df_processed


def preprocess_catcols(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    One-hot encode specified categorical features in a dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        categorical_cols (list): List of column names to one-hot encode.

    Returns:
        pd.DataFrame: A new dataframe with categorical columns one-hot encoded.
    """
    if not categorical_cols:  # safely skip if list is empty
        return df
    
    df_processed = df.copy()
    df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=False)
    return df_encoded


def balance_dataset(df: pd.DataFrame, label_column:str, strategy="downsample", random_state=42) -> pd.DataFrame:
    """
    Balances a DataFrame according to the class distribution in `label_column`.

    Args:
        df (pd.DataFrame): Input dataframe.
        label_column (str): Name of the column containing class labels.
        strategy (str): Either "downsample" or "upsample".
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: Balanced dataframe.
    """
    grouped = [group for _, group in df.groupby(label_column)]
    
    if strategy == "downsample":
        min_size = min(len(group) for group in grouped)
        balanced = [resample(group, replace=False, n_samples=min_size, random_state=random_state)
                    for group in grouped]
    elif strategy == "upsample":
        max_size = max(len(group) for group in grouped)
        balanced = [resample(group, replace=True, n_samples=max_size, random_state=random_state)
                    for group in grouped]
    else:
        raise ValueError("strategy must be 'downsample' or 'upsample'")
    
    return pd.concat(balanced).sample(frac=1, random_state=random_state).reset_index(drop=True) # type:ignore


def output_linear_classifier_features(examples, output_dir, dataset):
    def remove_constants(data):
        return data[[c for c in data if data[c].nunique() > 1]]
    # Normalize numerical variables analogously to LR, copied from fitted scaler in evaluate_external_dataset (seed 42).
    scalings = {
        'income': {'age': [38.66194047, 13.70079038], 'capital_gain': [1092.03493461, 7514.89341966],
                   'capital_loss': [87.05228675, 401.7001878], 'hours_per_week': [40.45123231, 12.43397048]},
        'car': {},
        'heart': {'Age': [53.63760218, 9.38893213], 'RestingBP': [132.09264305, 18.09209337],
                  'Cholesterol': [201.70844687, 107.50566557], 'FastingBS': [0.23160763, 0.42185962],
                  'MaxHR': [136.59945504, 25.12828773], 'Oldpeak': [0.92711172, 1.06128969]},
        'diabetes': {'Pregnancies': [3.68403909, 3.28025968], 'Glucose': [120.41042345, 32.63939221],
                     'BloodPressure': [68.75081433, 19.83518715], 'SkinThickness': [20.22638436, 15.68020872],
                     'Insulin': [79.43485342, 114.8289827], 'BMI': [31.77654723, 8.02507088],
                     'DiabetesPedigreeFunction': [0.47113192, 0.33090205], 'Age': [32.90879479, 11.66936554]}
    }
    scaling = scalings[dataset]

    def normalize_examples(data):
        for c in scaling.keys():
            data[c] = (data[c] - scaling[c][0]) / scaling[c][1]
        return data

    examples_dummies = remove_constants(pd.get_dummies(examples, dummy_na=True))

    if dataset == 'income':
        assert len(examples_dummies.columns) == 107

    # Also write out weighted version for linear explanation model
    examples_dummies = normalize_examples(examples_dummies)
    examples_dummies.to_pickle(output_dir / (dataset + '_lr_examples.p'))

    # Might be necessary for income
    # examples_dummies = remove_constants(pd.get_dummies(examples, dummy_na=True))

    # Sample examples for debugging
    # index_samples = np.random.choice(examples.index, min(200, len(examples)))
    # examples = examples.loc[index_samples]
    # examples_dummies = examples_dummies.loc[index_samples]


def create_perturbed_income_examples(examples, output_dir):
    num_perturbed_examples_per_example = 20
    prob_feature_perturbed = 2. / 12  # on average two features perturbed

    feature_values = {
        # Use max values for numerical features in training set with seed 42 (evaluate_external_dataset).
        'age': 99,
        'workclass': list(workclass_dict.keys()),
        'education': list(education_dict.keys()),
        'marital_status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                           'Married-spouse-absent', 'Married-AF-spouse'],
        'occupation': list(occupation_dict.keys()),
        'relationship': list(relationship_dict.keys()),
        'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
        'sex': ['Male', 'Female'],
        'capital_gain': 9999,
        'capital_loss': 4356,
        'hours_per_week': 99,
        'native_country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                           'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran',
                           'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal',
                           'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia',
                           'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador',
                           'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
    }
    def remove_constants(data):
        return data[[c for c in data if data[c].nunique() > 1]]
    examples_dummies = remove_constants(pd.get_dummies(examples, dummy_na=True))
    # Sample examples for debugging
    # index_samples = np.random.choice(examples.index, min(200, len(examples)))
    # examples = examples.loc[index_samples]
    # examples_dummies = examples_dummies.loc[index_samples]
    assert len(examples_dummies.columns) == 106
    # Normalize numerical variables analogously to LR, copied from fitted scaler in evaluate_external_dataset (seed 42).
    scaling = {
        'age': [38.66194047, 13.70079038],
        'capital_gain': [1092.03493461, 7514.89341966],
        'capital_loss': [87.05228675, 401.7001878],
        'hours_per_week': [40.45123231, 12.43397048]
    }
    def normalize_examples(data):
        for c in scaling.keys():
            data[c] = (data[c] - scaling[c][0]) / scaling[c][1]
        return data
    example_variants = examples.sample(0)
    for idx, ex in examples.iterrows():
        for p in range(0, num_perturbed_examples_per_example):
            perturbed_feature_mask = np.random.uniform(0, 1, 12) < prob_feature_perturbed
            example_copy = ex.copy()
            for f, feat in enumerate(list(feature_values.keys())):
                if perturbed_feature_mask[f]:
                    # Perturb this feature
                    if isinstance(feature_values[feat], int):
                        example_copy[feat] = int(random.uniform(0, feature_values[feat]))
                    else:
                        example_copy[feat] = feature_values[feat][int(random.uniform(0, len(feature_values[feat])))]
            # Store perturbed version for LLM inference
            example_variants = example_variants.append(example_copy, ignore_index=True)

    # Also write out weighted version for linear explanation model
    example_variants_dummies = pd.get_dummies(example_variants, dummy_na=True)
    for column in [c for c in examples_dummies.columns if c not in example_variants_dummies.columns]:
        example_variants_dummies[column] = 0
    example_variants_dummies = example_variants_dummies[examples_dummies.columns]
    examples_dummies = normalize_examples(examples_dummies)
    example_variants_dummies = normalize_examples(example_variants_dummies)
    counter = 0
    weights = []
    for idx, ex in examples_dummies.iterrows():
        for p in range(0, num_perturbed_examples_per_example):
            ex_original = [ex[c] for c in ex.index if c != 'label']
            ex_perturbed = example_variants_dummies.iloc[counter]
            ex_perturbed = [ex_perturbed[c] for c in ex_perturbed.index if c != 'label']
            weights.append(1 - spatial.distance.cosine(ex_original, ex_perturbed))
            counter += 1
    example_variants_dummies['weight'] = weights
    assert counter == example_variants_dummies.shape[0] and counter == example_variants.shape[0]
    print(f"Created {counter} perturbed examples.")
    example_variants_dummies.to_pickle(output_dir / 'weighted_perturbed_examples.p')
    return example_variants



