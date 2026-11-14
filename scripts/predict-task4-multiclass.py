import re
import joblib
import pyreadr
from bs4 import BeautifulSoup


# Same cleaning as PrimaryTaskMultiClass.ipynb (must match training)
def clean_html(html):
    soup = BeautifulSoup(str(html), "html.parser")
    text = " ".join(e.get_text(" ") for e in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]))
    text = re.sub(r"http\S+|www\S+|\S+@\S+", " ", text.lower())
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


claims_test = pyreadr.read_r("../data/claims-test.RData")["claims_test"]
model = joblib.load("../results/multiclass_model.joblib")

# Binary predictions come from predict-task4-binary.R; add multiclass alongside
pred_df = pyreadr.read_r("../results/task4-binary-predictions.RData")["binary_pred_df"]
assert (pred_df[".id"].values == claims_test[".id"].values).all()
pred_df["mclass.pred"] = model.predict(claims_test["text_tmp"].apply(clean_html))

pyreadr.write_rdata("../results/preds-group5.RData", pred_df, df_name="pred_df")
print(pred_df.head())
