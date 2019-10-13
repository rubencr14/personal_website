from sklearn.pipeline import Pipeline
from .Preprocessing import AttributeImputer, TransformCategoricalToNumerical, DatasetSplitter

def preprocessing_pipeline(df, path):

    pipeline = Pipeline(
                [("splitter", DatasetSplitter(PATH=path)),
                 ("imput", AttributeImputer()),
                 ("num", TransformCategoricalToNumerical())])

    return pipeline.fit_transform(df)

