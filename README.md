<h1>
<img src="./public/logo.png" width="30px" />
Predictive Tool of Organ/Space Surgical Site Infection
</h1>

# Linked
- Production: https://sonatatech.cn/osssi
- Pre-Production: https://osssi.vercel.app
- Development: https://osssi-dev.vercel.app

This repo developed by `onnxruntime-web` and `react`, you can clone this repo and run locally. `node` must be installed in your local computer.

# Run or Build

- Before Run or Build, you must install dependencies by `npm i`.

- And Run by `npm run dev`.

- And Build by `npm run build`.

**Pre-Production or Development** must run `npm run pre-build` to update `vercel.json`, and then add、commit、push.

## Models

The trained XGBoost models in [XGBoost Models](./src/models/xgb/)

# Future Works

- We will provide script for `TWA(*)` calculation.
- We will support offline batch risk prediction in `.excel` file and save risk results in `.excel` file.
- We are developing LLM based methods for predicting risk.