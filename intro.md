## 🔭 Introduction

**OnePower Explorer** is an interactive web application built on the **OnePower** Python package for computing cosmological large-scale structure observables using the halo model framework.

The app allows users to:

* Compute matter, galaxy-matter, and galaxy-galaxy power spectra
* Explore halo mass functions and stellar mass functions
* Investigate Halo Occupation Distribution (HOD) models
* Compare multiple cosmological and halo model configurations
* Analyse differences relative to a reference model

All calculations are performed live using the OnePower backend, with almost fully interactive visualisation.

---

## 🧭 How to Use the App

### 1️⃣ Choose Observables

In the **Observables** section of the sidebar:

* Select one or more quantities (e.g. Matter Power Spectrum, HMF, HOD)
* Optionally enable halo model components (1-halo / 2-halo)

Each selected observable appears in its own tab.

---

### 2️⃣ Adjust Cosmology

Under **Cosmological Parameters**, you can modify:

* $\Omega_{\mathrm{c}}$, $\Omega_{\mathrm{b}}$
* $h$, $n_{s}$, $\sigma_8$
* Redshift
* Neutrino mass
* Dark energy parameters ($w_0$, $w_a$)

All changes require the user to run the model again manuall by pressing the 🚀 Run model button.

---

### 3️⃣ Modify Halo Model Settings

Inside **Halo Model Parameters**, you can change:

* Mass function model
* Bias prescription
* Halo profile and concentration models
* Nonlinear corrections (HMCode, etc.)
* Overdensity definition

Advanced options are collapsible to keep the interface clean.

---

### 4️⃣ Configure HOD Models

In the **HOD Parameters** section:

* Choose between available HOD models
* Adjust central and satellite occupation parameters
* Explore their impact on galaxy power spectra

---

### 5️⃣ Compare Models

You can:

* Save the current configuration as a comparison model
* Clear saved models
* Set a reference model
* Enable ratio comparison

This allows direct visual comparison between cosmological or halo model choices.

---

### 6️⃣ Download Data

Each tab allows you to download the displayed data as a CSV file for further analysis.

---

## ⚠️ Important Notice

This web app provides a convenient interactive interface for exploring cosmological predictions, but it does **not** expose the full flexibility and capability of the underlying OnePower package.

For:

* Full control of model configurations
* Batch computations
* Pipeline integration
* Scientific production use
* Detailed documentation

please visit the official OnePower GitHub repository and documentation.

The web interface is intended for exploration, testing, visual intuition, and rapid prototyping.

---

## 🎓 Who Is This For?

* Cosmologists exploring halo model behaviour
* Students learning large-scale structure theory
* Researchers testing parameter sensitivity
* Anyone curious about how cosmology shapes power spectra
