import { useState } from "react";
import axios from "axios";

export default function PredictionForm() {
  const [form, setForm] = useState({
    GrLivArea: 1500,
    OverallQual: 7,
    YearBuilt: 2005,
    GarageCars: 2,
    TotalBsmtSF: 900,
    FullBath: 2,
  });
  const [price, setPrice] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    const numericFields = ["GrLivArea", "OverallQual", "YearBuilt", "GarageCars", "TotalBsmtSF", "FullBath"];
    setForm({
        ...form,
        [name]: numericFields.includes(name) ? Number(value) : value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post("http://127.0.0.1:8000/predict", form);
      setPrice(res.data.predicted_price.toFixed(2));
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="bg-white shadow-xl p-6 rounded-2xl w-96">
      <h2 className="text-xl font-semibold mb-4">Predict House Price</h2>
      <form onSubmit={handleSubmit} className="space-y-3">
        {Object.keys(form).map((key) => (
          <div key={key}>
            <label className="block text-sm font-medium">{key}</label>
            <input
              type="number"
              name={key}
              value={form[key]}
              onChange={handleChange}
              className="w-full border rounded p-2"
            />
          </div>
        ))}
        <button
          type="submit"
          className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
        >
          Predict
        </button>
      </form>
      {price && (
        <div className="mt-4 text-center text-lg font-semibold text-green-600">
          Predicted Price: ${price}
        </div>
      )}
    </div>
  );
}
