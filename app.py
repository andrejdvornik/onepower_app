import streamlit as st
import numpy as np
import io
import plotly.graph_objects as go
import plotly.colors as pc
import plotly.io as pio
from pathlib import Path
import hashlib
import json
from onepower import Spectra
from pk_to_real import PkTransformer

with Path.open('load_mathjax.js') as f:
    js = f.read()
    st.components.v1.html(f'<script>{js}</script>', height=0)


OBSERVABLE_MAP = {
    r'Matter Power Spectrum $P_{\mathrm{mm}}(k)$': ('pk', 'mm'),
    r'Galaxy-matter Power Spectrum $P_{\mathrm{gm}}(k)$': ('pk', 'gm'),
    r'Galaxy-galaxy Power Spectrum $P_{\mathrm{gg}}(k)$': ('pk', 'gg'),
    r'Intrinsic-intrinsic Power Spectrum $P_{\mathrm{II}}(k)$': ('pk', 'ii'),
    r'Galaxy-Intrinsic Power Spectrum $P_{\mathrm{gI}}(k)$': ('pk', 'gi'),
    r'Matter-Intrinsic Power Spectrum $P_{\mathrm{mI}}(k)$': ('pk', 'mi'),
    r'Galaxy Bias $b_{\mathrm{g}}(k)$': ('pk', 'gb'),
    'Halo Mass Function': ('mass', 'hmf'),
    'Halo Bias Function': ('mass', 'bias'),
    'Concentration (matter)': ('mass', 'conc_cen'),
    'Concentration (galaxies)': ('mass', 'conc_sat'),
    'Stellar Mass Function': ('mass', 'smf'),
    'HOD': ('mass', 'hod'),
    r'$\Delta \Sigma (r_{\mathrm{p}})$': ('proj', 'ds'),
    r'$w_{\mathrm{p}}(r_{\mathrm{p}})$': ('proj', 'wp'),
    r'$w(\theta)$': ('proj', 'wtheta'),
    r'$\gamma_{\mathrm{t}}(\theta)$': ('proj', 'gamma'),
    r'$\xi_{+} (\theta)$': ('proj', 'xip'),
    r'$\xi_{-} (\theta)$': ('proj', 'xim'),
}


def get_streamlit_theme():

    return {
        'primary': st.get_option('theme.primaryColor'),
        'background': st.get_option('theme.backgroundColor'),
        'secondary_bg': st.get_option('theme.secondaryBackgroundColor'),
        'text': st.get_option('theme.textColor'),
    }


def set_plotly_theme_from_streamlit():

    theme = get_streamlit_theme()

    pio.templates['streamlit_matplotlib'] = go.layout.Template(
        layout=go.Layout(
            font=dict(family='Times New Roman', size=14, color=theme['text']),
            plot_bgcolor=theme['background'],
            paper_bgcolor=theme['background'],
            colorway=[
                theme['primary'],
                theme['text'],
                '#888888',
                '#555555',
            ],
            xaxis=dict(
                showline=True,
                linewidth=1.5,
                linecolor=theme['text'],
                mirror=True,
                ticks='inside',
                tickwidth=1.2,
                tickcolor=theme['text'],
                showgrid=False,
            ),
            yaxis=dict(
                showline=True,
                linewidth=1.5,
                linecolor=theme['text'],
                mirror=True,
                ticks='inside',
                tickwidth=1.2,
                tickcolor=theme['text'],
                showgrid=False,
            ),
            legend=dict(
                borderwidth=0, bgcolor='rgba(0,0,0,0)', font=dict(color=theme['text'])
            ),
        )
    )

    pio.templates.default = 'streamlit_matplotlib'


def plot_observable(
    x, y_dict, name, compare_reference, components=False, logx=True, logy=True
):

    theme = get_streamlit_theme()

    fig = go.Figure()

    # Plotly default palette (skip first color)
    plotly_colors = pc.qualitative.Plotly[1:]

    component_keys = [k for k in y_dict if k != 'tot']

    if components:
        for i, comp in enumerate(component_keys):
            color = plotly_colors[i % len(plotly_colors)]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_dict[comp] if name != 'mi' else np.abs(y_dict[comp]),
                    mode='lines',
                    name=comp,
                    line=dict(color=color, width=2),
                )
            )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_dict['tot'] if name != 'mi' else np.abs(y_dict['tot']),
            mode='lines',
            name='Live Model',
            line=dict(color=theme['primary'], width=3),
        )
    )

    offset = len(component_keys)
    for i, model in enumerate(st.session_state.models):
        if name in model['outputs']:
            x_stored, y_stored = model['outputs'][name]
            color = plotly_colors[(i + offset) % len(plotly_colors)]
            fig.add_trace(
                go.Scatter(
                    x=x_stored,
                    y=y_stored['tot'] if name != 'mi' else np.abs(y_stored['tot']),
                    mode='lines',
                    name=f'Model {i + 1}',
                    line=dict(color=color, width=2),
                )
            )

    if compare_reference and st.session_state.reference_model is not None:
        model_ref = st.session_state.reference_model
        if name in model_ref['outputs']:
            x_stored, y_stored = model_ref['outputs'][name]
            color = plotly_colors[(offset) % len(plotly_colors)]
            fig.add_trace(
                go.Scatter(
                    x=x_stored,
                    y=y_stored['tot'] if name != 'mi' else np.abs(y_stored['tot']),
                    mode='lines',
                    name='Reference model',
                    line=dict(color=color, width=2),
                )
            )

    # ---- Axis formatting ----
    if name in [
        'mm',
        'gm',
        'gg',
        'ii',
        'gi',
        'hmf',
        'smf',
        'bias',
        'gb',
        'conc_cen',
        'conc_sat',
        'ds',
        'wp',
        'wtheta',
        'gamma',
        'xip',
        'xim',
    ]:
        y_range = [np.log10(np.min(y['tot']) * 0.5), np.log10(np.max(y['tot']) * 2)]
    elif name == 'mi':
        y_range = [
            np.log10(np.min(np.abs(y['tot'])) * 0.5),
            np.log10(np.max(np.abs(y['tot'])) * 2),
        ]
    elif name in ['hod']:
        y_range = [
            np.max([np.log10(np.min(y['tot']) * 0.5), -3]),
            np.min([np.log10(np.max(y['tot']) * 2), 5]),
        ]
    fig.update_layout(
        xaxis_type='log' if logx else 'linear',
        yaxis_type='log' if logy else 'linear',
        yaxis_range=y_range,
        width=700,
        height=460,
        margin=dict(l=60, r=20, t=40, b=60),
        # font=dict(family="STIXGeneral")
    )
    fig.update_traces(
        hovertemplate='x = %{x:.3e}<br>y = %{y:.3e}<extra></extra>',
        showlegend=True,
    )

    # ---- Scientific axis labels ----
    if name in ['mm', 'gm', 'gg', 'ii', 'gi']:
        fig.update_xaxes(title=r'$k\,[h\,\mathrm{Mpc}^{-1}]$')
        fig.update_yaxes(title=r'$P(k)\,[(\mathrm{Mpc}/h)^3]$')
    elif name == 'mi':
        fig.update_xaxes(title=r'$k\,[h\,\mathrm{Mpc}^{-1}]$')
        fig.update_yaxes(title=r'$|P(k)|\,[(\mathrm{Mpc}/h)^3]$')
    elif name == 'gb':
        fig.update_xaxes(title=r'$k\,[h\,\mathrm{Mpc}^{-1}]$')
        fig.update_yaxes(title=r'$b_{\mathrm{g}}(k)$')
    elif name == 'wp':
        fig.update_xaxes(title=r'$r_{\mathrm{p}}\,[h^{-1}\,\mathrm{Mpc}]$')
        fig.update_yaxes(
            title=r'$w_{\mathrm{p}}(r_{\mathrm{p}})\,[h^{-1}\,\mathrm{Mpc}]$'
        )
    elif name == 'ds':
        fig.update_xaxes(title=r'$r_{\mathrm{p}}\,[h^{-1}\,\mathrm{Mpc}]$')
        fig.update_yaxes(title=r'$\Delta \Sigma\,[hM_{\odot}/\mathrm{pc}^2]$')
    elif name == 'wtheta':
        fig.update_xaxes(title=r'$\theta\,[\mathrm{arcmin}]$')
        fig.update_yaxes(title=r'$w(\theta)$')
    elif name == 'gamma':
        fig.update_xaxes(title=r'$\theta\,[\mathrm{arcmin}]$')
        fig.update_yaxes(title=r'$\gamma_{\mathrm{t}}(\theta)$')
    elif name == 'xip' or name == 'xim':
        fig.update_xaxes(title=r'$\theta\,[\mathrm{arcmin}]$')
        fig.update_yaxes(title=r'$\xi_{+}(\theta)$')
    elif name == 'hod':
        fig.update_xaxes(title=r'$M^{*}\,[h^{-2}M_{\odot}]$')
        fig.update_yaxes(title=r'$<N|M>$')
    elif name == 'smf':
        fig.update_xaxes(title=r'$M^{*}\,[h^{-2}M_{\odot}]$')
        fig.update_yaxes(title=r'$\Phi\,[h^{3} \mathrm{dex}^{-1} \mathrm{Mpc}^{-3}]$')
    elif name == 'hmf':
        fig.update_xaxes(title=r'$M_{h}\,[h^{-1}M_{\odot}]$')
        fig.update_yaxes(title=r'$\mathrm{d}n / \mathrm{d} \mathrm{ln} M$')
    elif name == 'bias':
        fig.update_xaxes(title=r'$M_{h}\,[h^{-1}M_{\odot}]$')
        fig.update_yaxes(title=r'$b_{h}(M)$')
    elif name == 'conc_cen' or name == 'conc_sat':
        fig.update_xaxes(title=r'$M_{h}\,[h^{-1}M_{\odot}]$')
        fig.update_yaxes(title=r'$c(M)$')
    fig.update_xaxes(exponentformat='power')
    fig.update_yaxes(exponentformat='power')

    return fig


def plot_ratio(x, y_live, x_ref, y_ref, name, logx=True):
    theme = get_streamlit_theme()
    y_ref = np.interp(x, x_ref, y_ref['tot'])
    ratio = ((y_live['tot'] - y_ref) / y_ref) * 100.0

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=ratio,
            mode='lines',
            name='(Live - Ref) / Ref',
            line=dict(color=theme['primary'], width=3),
        )
    )

    fig.update_layout(
        xaxis_type='log' if logx else 'linear',
        yaxis_type='linear',
        yaxis_title=r'$\mathrm{Relative\ difference\ } [\%]$',
        yaxis_range=[-np.max(np.absolute(ratio)), np.max(np.absolute(ratio))],
        width=700,
        height=460,
        margin=dict(l=60, r=20, t=40, b=60),
    )
    if name in ['mm', 'gm', 'gg', 'ii', 'gi', 'mi', 'gb']:
        fig.update_xaxes(title=r'$k\,[h\,\mathrm{Mpc}^{-1}]$')
    elif name == 'wp' or name == 'ds':
        fig.update_xaxes(title=r'$r_{\mathrm{p}}\,[h^{-1}\,\mathrm{Mpc}]$')
    elif name in ['wtheta', 'gamma', 'xip', 'xim']:
        fig.update_xaxes(title=r'$\theta\,[\mathrm{arcmin}]$')
    elif name in ['hmf', 'conc_cen', 'conc_sat', 'bias']:
        fig.update_xaxes(title=r'$M_{h}\,[h^{-1}M_{\odot}]$')
    elif name in ['hod', 'smf']:
        fig.update_xaxes(title=r'$M^{*}\,[h^{-2}M_{\odot}]$')

    # Add horizontal unity line
    fig.add_hline(y=0.0, line_width=1)

    fig.update_traces(
        hovertemplate='x = %{x:.3e}<br>y = %{y:.3e}<extra></extra>',
        showlegend=True,
    )
    fig.update_xaxes(exponentformat='power')

    return fig


def compute_power_spectrum(model, spectrum_type, components=False):
    ps_attr = {
        'mm': 'power_spectrum_mm',
        'gm': 'power_spectrum_gm',
        'gg': 'power_spectrum_gg',
        'ii': 'power_spectrum_ii',
        'gi': 'power_spectrum_gi',
        'mi': 'power_spectrum_mi',
        'gb': 'power_spectrum_gm',
    }

    ps = getattr(model, ps_attr[spectrum_type])

    k = model.k_vec
    if spectrum_type == 'gb':
        return k, {'tot': ps.galaxy_linear_bias[0, 0, :]}

    pk_tot = ps.pk_tot[0, 0, :]

    if components:
        pk_1h = ps.pk_1h[0, 0, :]
        pk_2h = ps.pk_2h[0, 0, :]
        return k, {'tot': pk_tot, '1h': pk_1h, '2h': pk_2h}

    return k, {'tot': pk_tot}


def compute_proj(model, corr_type, rpmin, rpmax, thetamin, thetamax, components=False):
    if corr_type in ['ds', 'wp']:
        sep_min_in = rpmin
        sep_max_in = rpmax
    elif corr_type in ['wtheta', 'gamma', 'xip', 'xim']:
        sep_min_in = thetamin
        sep_max_in = thetamax
    else:
        sep_min_in = None
        sep_max_in = None
    transformer = PkTransformer(
        corr_type,
        model,
        sep_min_in=sep_min_in,
        sep_max_in=sep_max_in,
        components=components,
    )
    if components:
        sep, xi, xi_1h, xi_2h = transformer()
        return sep, {'tot': xi, '1h': xi_1h, '2h': xi_2h}
    else:
        sep, xi = transformer()
        return sep, {'tot': xi}


def compute_mass_quantity(model, quantity, components=False):
    if quantity == 'hmf':
        return model.mass, {'tot': model.dndlnm[0, :]}

    if quantity == 'smf':
        fail_obs_func = np.logspace(8.0, 12.0, 300)
        if components:
            return model.obs_func_obs[
                0, 0, :
            ] if model.obs_func is not None else fail_obs_func, {
                'tot': model.obs_func[0, 0, :]
                if model.obs_func is not None
                else np.zeros(300),
                'cen': model.obs_func_cen[0, 0, :]
                if model.obs_func is not None
                else np.zeros(300),
                'sat': model.obs_func_sat[0, 0, :]
                if model.obs_func is not None
                else np.zeros(300),
            }
        return model.obs_func_obs[
            0, 0, :
        ] if model.obs_func is not None else fail_obs_func, {
            'tot': model.obs_func[0, 0, :]
            if model.obs_func is not None
            else np.zeros(300)
        }

    if quantity == 'hod':
        if components:
            return model.mass, {
                'tot': model.hod.hod[0, 0, :],
                'cen': model.hod.hod_cen[0, 0, :],
                'sat': model.hod.hod_sat[0, 0, :],
            }
        return model.mass, {'tot': model.hod.hod[0, 0, :]}

    if quantity == 'bias':
        return model.mass, {'tot': model.halo_bias[0, :]}

    if quantity == 'conc_cen':
        return model.mass, {'tot': model.conc_cen[0, :]}

    if quantity == 'conc_sat':
        return model.mass, {'tot': model.conc_sat[0, :]}


def hash_params(params):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    params_serializable = convert(params)
    params_string = json.dumps(params_serializable, sort_keys=True)
    return hashlib.md5(params_string.encode()).hexdigest()


@st.cache_resource(show_spinner=False, ttl=1800)
def compute_outputs(params, components=True):
    rpmin = params['rpmin']
    rpmax = params['rpmax']
    thetamin = params['thetamin']
    thetamax = params['thetamax']
    del params['rpmin']
    del params['rpmax']
    del params['thetamin']
    del params['thetamax']

    st.session_state.init_model.update(**params)
    model = st.session_state.init_model

    computed_outputs = {}
    for output in OBSERVABLE_MAP:
        category, subtype = OBSERVABLE_MAP[output]
        if category == 'pk':
            x, y = compute_power_spectrum(model, subtype, components)
            computed_outputs[subtype] = (x, y)
        elif category == 'mass':
            x, y = compute_mass_quantity(model, subtype, components)
            computed_outputs[subtype] = (x, y)
        elif category == 'proj':
            x, y = compute_proj(
                model, subtype, rpmin, rpmax, thetamin, thetamax, components
            )
            computed_outputs[subtype] = (x, y)
    return computed_outputs


if __name__ == '__main__':
    set_plotly_theme_from_streamlit()

    st.set_page_config(layout='wide', page_title='OnePower Explorer')

    st.image(
        'https://andrej.dvrnk.si/page/wp-content/uploads/2025/08/logosmall_black_merged.png',
        width=500,
    )
    st.title('The OnePower Explorer')
    st.text('The One App to Explore the Halo Model and its Predictions.')
    # st.divider()

    # --------------------------
    # Session state for models
    # --------------------------
    st.session_state.init_model = Spectra()
    if 'models' not in st.session_state:
        st.session_state.models = []

    if 'reference_model' not in st.session_state:
        st.session_state.reference_model = None

    if 'computed_outputs' not in st.session_state:
        st.session_state.computed_outputs = None

    if 'has_run' not in st.session_state:
        st.session_state.has_run = False

    if 'params_hash' not in st.session_state:
        st.session_state.params_hash = None

    if 'selected_outputs' not in st.session_state:
        st.session_state.selected_outputs = [
            r'Matter Power Spectrum $P_{\mathrm{mm}}(k)$'
        ]

    st.sidebar.link_button(
        '💾 OnePower PyPi',
        'https://pypi.org/project/onepower/',
        width='stretch',
    )
    st.sidebar.link_button(
        '📦 OnePower GitHub',
        'https://github.com/KiDS-WL/onepower',
        width='stretch',
    )
    st.sidebar.header('Input Parameters and Settings')

    compare_reference = st.sidebar.toggle('Compare to reference model', False)
    components = st.sidebar.toggle('Show individual halo model components', value=False)

    with st.sidebar.expander('Quantities', expanded=True):
        selected_outputs = []
        for label in OBSERVABLE_MAP:
            checked = st.checkbox(
                label,
                value=label in st.session_state.selected_outputs,
                key=f'chk_{label}',
            )
            if checked:
                selected_outputs.append(label)
        st.session_state.selected_outputs = selected_outputs

    no_selection = len(st.session_state.selected_outputs) == 0
    if no_selection:
        st.error('Please select at least one observable to compute.', icon='⚠️')

    with st.form('parameter_form', width=500):
        run_model = st.form_submit_button(
            '🚀 Run model',
            width='stretch',
        )
        with st.sidebar:
            with st.sidebar.expander('General settings', expanded=False):
                kmin = st.number_input(
                    r'$k_{\mathrm{min}}\,[h\,\mathrm{Mpc}^{-1}]$',
                    value=1e-3,
                    format='%.4e',
                )
                kmax = st.number_input(
                    r'$k_{\mathrm{max}}\,[h\,\mathrm{Mpc}^{-1}]$', value=10.0
                )
                nk = st.number_input(r'Number of $k$ points', 10, 1000, 300)

                k_vec = np.logspace(np.log10(kmin), np.log10(kmax), nk)

                mmin = st.number_input(
                    r'$M_{h,\mathrm{min}}\,[h^{-1}\,M_{\odot}]$', value=9.0
                )
                mmax = st.number_input(
                    r'$M_{h,\mathrm{max}}\,[h^{-1}\,M_{\odot}]$', value=15.0
                )
                # nm = st.number_input("Number of M points", 10, 1000, 300)

                rpmin = st.number_input(
                    r'$r_{\mathrm{p, min}}\,[h^{-1}\,\mathrm{Mpc}]$', value=0.1
                )
                rpmax = st.number_input(
                    r'$r_{\mathrm{p, max}}\,[h^{-1}\,\mathrm{Mpc}]$', value=20.0
                )

                thetamin = st.number_input(
                    r'$\theta_{\mathrm{min}}\,[\mathrm{arcmin}]$', value=0.5
                )
                thetamax = st.number_input(
                    r'$\theta_{\mathrm{max}}\,[\mathrm{arcmin}]$', value=200.0
                )

            with st.sidebar.expander('Cosmological Parameters', expanded=False):
                omega_c = st.number_input(r'$\Omega_{c}$', 0.1, 0.5, 0.25, 0.01)
                omega_b = st.number_input(r'$\Omega_{b}$', 0.02, 0.08, 0.05, 0.005)
                h = st.number_input(r'$h$', 0.5, 0.9, 0.7, 0.01)
                ns = st.number_input(r'$n_s$', 0.8, 1.2, 0.9, 0.005)
                sigma_8 = st.number_input(r'$\sigma_8$', 0.6, 1.0, 0.8, 0.01)
                z_vec = st.number_input(r'Redshift $z$', 0.0, 2.0, 0.0, 0.1)
                m_nu = st.number_input(
                    r'Sum of Neutrino Masses $m_{\nu} [eV]$', 0.0, 1.0, 0.06, 0.01
                )
                w0 = st.number_input(
                    r'Dark Energy Equation of State $w_0$', -1.5, -0.5, -1.0, 0.05
                )
                wa = st.number_input(
                    r'Dark Energy Equation of State $w_a$', 0.0, 1.0, 0.0, 0.05
                )
                tcmb = st.number_input(
                    r'CMB Temperature $T_{\mathrm{cmb}} [K]$', 2.0, 3.0, 2.7255, 0.01
                )

            with st.sidebar.expander('Halo Model Parameters', expanded=False):
                dewiggle = st.toggle('Dewiggle', value=False)
                pointmass = st.toggle('Point Mass', value=False)
                # response = st.toggle("Response", value=False)
                response = False
                mdef_model = st.selectbox(
                    'Mass definition model',
                    ('SOMean', 'SOVirial', 'SOCritical', 'FOF'),
                )
                hmf_model = st.selectbox(
                    'Halo mass function model',
                    (
                        'Tinker10',
                        'ST',
                        'PS',
                        'SMT',
                        'Jenkins',
                        'Warren',
                        'Reed03',
                        'Reed07',
                        'Peacock',
                        'Angulo',
                        'AnguloBound',
                        'Watson',
                        'Watson_FoF',
                        'Crocce',
                        'Courtin',
                        'Bhattacharya',
                        'Tinker08',
                        'Behroozi',
                        'Pillepich',
                        'Manera',
                        'Ishiyama',
                        'Bocquet200mDMOnly',
                        'Bocquet200mHydro',
                        'Bocquet200cDMOnly',
                        'Bocquet200cHydro',
                        'Bocquet500cDMOnly',
                        'Bocquet500cHydro',
                    ),
                )
                bias_model = st.selectbox(
                    'Halo bias function model',
                    (
                        'Tinker10',
                        'Tinker10PBSplit',
                        'ST99',
                        'Mo96',
                        'Jing98',
                        'SMT01',
                        'Seljak04',
                        'Seljak04Cosmo',
                        'Tinker05',
                        'Mandelbaum05',
                        'Pillepich10',
                        'Manera10',
                        'TinkerSD05',
                    ),
                )
                halo_profile_model_dm = st.selectbox(
                    'Halo profile model (matter)',
                    (
                        'NFW',
                        'NFWInf',
                        'GeneralizedNFW',
                        'GeneralizedNFWInf',
                        'Einasto',
                        'Hernquist',
                        'HernquistInf',
                        'Moore',
                        'MooreInf',
                        'Constant',
                        'CoreNFW',
                        'PowerLawWithExpCut',
                    ),
                )
                halo_profile_model_sat = st.selectbox(
                    'Halo profile model (galaxies)',
                    (
                        'NFW',
                        'NFWInf',
                        'GeneralizedNFW',
                        'GeneralizedNFWInf',
                        'Einasto',
                        'Hernquist',
                        'HernquistInf',
                        'Moore',
                        'MooreInf',
                        'Constant',
                        'CoreNFW',
                        'PowerLawWithExpCut',
                    ),
                )
                halo_concentration_model_dm = st.selectbox(
                    'Halo concentration model (matter)',
                    (
                        'Duffy08',
                        'Bullock01',
                        'Bullock01Power',
                        'Maccio07',
                        'Zehavi11',
                        'Ludlow16',
                        'Ludlow16Empirical',
                    ),
                )
                halo_concentration_model_sat = st.selectbox(
                    'Halo concentration model (galaxies)',
                    (
                        'Duffy08',
                        'Bullock01',
                        'Bullock01Power',
                        'Maccio07',
                        'Zehavi11',
                        'Ludlow16',
                        'Ludlow16Empirical',
                    ),
                )
                overdensity = st.number_input(
                    'Halo overdensity', 0.0, 500.0, 200.0, 1.0
                )
                delta_c = st.number_input(
                    r'Collapse threshold $\delta_c$',
                    0.0,
                    10.0,
                    1.696,
                    0.001,
                    format='%0.3f',
                )
                norm_cen = st.number_input(
                    r'Normalization of $c(M)$ relation (matter)', 0.0, 2.0, 1.0, 0.01
                )
                norm_sat = st.number_input(
                    r'Normalization of $c(M)$ relation (galaxies)', 0.0, 2.0, 1.0, 0.01
                )
                eta_cen = st.number_input(
                    r'Halo bloating $\eta$ (matter)', -1.0, 1.0, 0.0, 0.01
                )
                eta_sat = st.number_input(
                    r'Halo bloating $\eta$ (galaxies)', -1.0, 1.0, 0.0, 0.01
                )

                hmcode_ingredients = st.selectbox(
                    'HMCode ingredients', [None, 'mead2020', 'mead2020_feedback', 'fit']
                )
                if hmcode_ingredients == 'mead2020_feedback':
                    log10T_AGN = st.number_input(
                        r'$\log_{10}T_{\mathrm{AGN}}$', 0.0, 10.0, 7.8, 0.01
                    )
                else:
                    log10T_AGN = 7.8

                if hmcode_ingredients == 'fit':
                    mb = st.number_input(r'$M_b$', 8.0, 15.0, 13.87, 0.01)
                else:
                    mb = 13.87

                nonlinear_mode = st.selectbox(
                    'Nonlinear mode', [None, 'bnl', 'hmcode', 'fortuna']
                )
                if nonlinear_mode == 'fortuna':
                    t_eff = st.number_input(r'$t_{\mathrm{eff}}$', 0.0, 1.0, 0.0, 0.01)
                else:
                    t_eff = 0.0

            with st.sidebar.expander('HOD Parameters', expanded=False):
                hod_model = st.selectbox(
                    'HOD model',
                    ('Cacciato', 'Zheng', 'Simple', 'Zehavi', 'Zhai'),
                )
                obs_min = st.number_input(
                    r'Min Observable Mass $[h^{-2}\,M_{\odot}]$', 8.0, 15.0, 8.0, 0.1
                )
                obs_max = st.number_input(
                    r'Max Observable Mass $[h^{-2}\,M_{\odot}]$', 8.0, 15.0, 12.0, 0.1
                )
                hod_settings = {
                    'observables_file': None,
                    'zmin': np.array([0.0]),
                    'zmax': np.array([2.0]),
                    'obs_min': np.array([obs_min]),
                    'obs_max': np.array([obs_max]),
                    'nz': 15,
                    'nobs': 300,
                    'observable_h_unit': '1/h^2',
                }
                obs_settings = {
                    'observables_file': None,
                    'zmin': np.array([z_vec]),
                    'zmax': np.array([z_vec]),
                    'obs_min': np.array([8.0]),
                    'obs_max': np.array([12.0]),
                    'nz': 1,
                    'nobs': 300,
                    'observable_h_unit': '1/h^2',
                }
                if hod_model == 'Cacciato':
                    compute_observable = True
                    log10_obs_norm_c = st.number_input(
                        r'$\log_{10} O_{\mathrm{norm, c}}$', value=9.95
                    )
                    log10_m_ch = st.number_input(
                        r'$\log_{10} M_{\mathcal{ch}}$', value=11.24
                    )
                    g1 = st.number_input(r'$\gamma_1$', value=3.18)
                    g2 = st.number_input(r'$\gamma_2$', value=0.245)
                    sigma_log10_O_c = st.number_input(
                        r'$\sigma_{\mathrm{c}}$', value=0.157
                    )
                    norm_s = st.number_input(
                        r'$\mathrm{norm}_{\mathrm{s}}$', value=0.562
                    )
                    pivot = st.number_input(r'$M_{\mathrm{pivot}}$', value=12.0)
                    alpha_s = st.number_input(r'$\alpha_{\mathrm{s}}$', value=-1.18)
                    beta_s = st.number_input(r'$\beta_{\mathrm{s}}$', value=2.0)
                    b0 = st.number_input(r'$b_0$', value=-1.17)
                    b1 = st.number_input(r'$b_1$', value=1.53)
                    b2 = st.number_input(r'$b_2$', value=-0.217)
                    A_cen = st.number_input(
                        r'Assembly bias parameter $A_{\mathrm{cen}}$',
                        -1.0,
                        1.0,
                        0.0,
                        0.01,
                    )
                    A_sat = st.number_input(
                        r'Assembly bias parameter $A_{\mathrm{sat}}$',
                        -1.0,
                        1.0,
                        0.0,
                        0.01,
                    )
                    hod_params = {
                        'log10_obs_norm_c': log10_obs_norm_c,
                        'log10_m_ch': log10_m_ch,
                        'g1': g1,
                        'g2': g2,
                        'sigma_log10_O_c': sigma_log10_O_c,
                        'norm_s': norm_s,
                        'pivot': pivot,
                        'alpha_s': alpha_s,
                        'beta_s': beta_s,
                        'b0': b0,
                        'b1': b1,
                        'b2': b2,
                        'A_cen': A_cen,
                        'A_sat': A_sat,
                    }

                if hod_model == 'Zheng':
                    compute_observable = False
                    log10_Mmin = st.number_input(
                        r'$\log_{10}M_{\mathrm{min}}$', value=12.0
                    )
                    log10_M0 = st.number_input(r'$\log_{10}M_{0}$', value=12.0)
                    log10_M1 = st.number_input(r'$\log_{10}M_{1}$', value=13.0)
                    sigma = st.number_input(r'$\sigma$', value=0.15)
                    alpha = st.number_input(r'$\alpha$', value=1.0)
                    A_cen = st.number_input(
                        r'Assembly bias parameter $A_{\mathrm{cen}}$',
                        -1.0,
                        1.0,
                        0.0,
                        0.01,
                    )
                    A_sat = st.number_input(
                        r'Assembly bias parameter $A_{\mathrm{sat}}$',
                        -1.0,
                        1.0,
                        0.0,
                        0.01,
                    )
                    hod_params = {
                        'log10_Mmin': log10_Mmin,
                        'log10_M0': log10_M0,
                        'log10_M1': log10_M1,
                        'sigma': sigma,
                        'alpha': alpha,
                        'A_cen': A_cen,
                        'A_sat': A_sat,
                    }
                if hod_model == 'Simple':
                    compute_observable = False
                    log10_Mmin = st.number_input(
                        r'$\log_{10}M_{\mathrm{min}}$', value=12.0
                    )
                    log10_Msat = st.number_input(
                        r'$\log_{10}M_{\mathrm{sat}}$', value=13.0
                    )
                    alpha = st.number_input(r'$\alpha$', value=1.0)
                    A_cen = st.number_input(
                        r'Assembly bias parameter $A_{\mathrm{cen}}$',
                        -1.0,
                        1.0,
                        0.0,
                        0.01,
                    )
                    A_sat = st.number_input(
                        r'Assembly bias parameter $A_{\mathrm{sat}}$',
                        -1.0,
                        1.0,
                        0.0,
                        0.01,
                    )
                    hod_params = {
                        'log10_Mmin': log10_Mmin,
                        'log10_Msat': log10_Msat,
                        'alpha': alpha,
                        'A_cen': A_cen,
                        'A_sat': A_sat,
                    }
                if hod_model == 'Zehavi':
                    compute_observable = False
                    log10_Mmin = st.number_input(
                        r'$\log_{10}M_{\mathrm{min}}$', value=12.0
                    )
                    log10_Msat = st.number_input(
                        r'$\log_{10}M_{\mathrm{sat}}$', value=13.0
                    )
                    alpha = st.number_input(r'$\alpha$', value=1.0)
                    A_cen = st.number_input(
                        r'Assembly bias parameter $A_{\mathrm{cen}}$',
                        -1.0,
                        1.0,
                        0.0,
                        0.01,
                    )
                    A_sat = st.number_input(
                        r'Assembly bias parameter $A_{\mathrm{sat}}$',
                        -1.0,
                        1.0,
                        0.0,
                        0.01,
                    )
                    hod_params = {
                        'log10_Mmin': log10_Mmin,
                        'log10_Msat': log10_Msat,
                        'alpha': alpha,
                        'A_cen': A_cen,
                        'A_sat': A_sat,
                    }
                if hod_model == 'Zhai':
                    compute_observable = False
                    log10_Mmin = st.number_input(
                        r'$\log_{10}M_{\mathrm{min}}$', value=13.58
                    )
                    log10_Msat = st.number_input(
                        r'$\log_{10}M_{\mathrm{sat}}$', value=14.87
                    )
                    log10_Mcut = st.number_input(
                        r'$\log_{10}M_{\mathrm{cut}}$', value=12.32
                    )
                    sigma = st.number_input(r'$\sigma$', value=0.82)
                    alpha = st.number_input(r'$\alpha$', value=0.41)
                    A_cen = st.number_input(
                        r'Assembly bias parameter $A_{\mathrm{cen}}$',
                        -1.0,
                        1.0,
                        0.0,
                        0.01,
                    )
                    A_sat = st.number_input(
                        r'Assembly bias parameter $A_{\mathrm{sat}}$',
                        -1.0,
                        1.0,
                        0.0,
                        0.01,
                    )
                    hod_params = {
                        'log10_Mmin': log10_Mmin,
                        'log10_Msat': log10_Msat,
                        'log10_Mcut': log10_Mcut,
                        'sigma': sigma,
                        'alpha': alpha,
                        'A_cen': A_cen,
                        'A_sat': A_sat,
                    }

            with st.sidebar.expander('IA Parameters', expanded=False):
                st.warning(
                    'The IA parameters are currently fixed, but they will be included in future updates of the app. The only option that is currently avaiable is to show the fixed     IA  power spectra, using Fortuna et al. 2021 model.',
                    icon='⚠️',
                )

    if 'Stellar Mass Function' in selected_outputs and hod_model != 'Cacciato':
        st.warning(
            f'The Stellar Mass Function cannot be computed with the {hod_model} HOD model, since it does not include an explicit observable-mass relation. Please switch to the Cacciato HOD model to compute the SMF.',
            icon='⚠️',
        )

    params = {
        'omega_c': omega_c,
        'omega_b': omega_b,
        'h0': h,
        'n_s': ns,
        'sigma_8': sigma_8,
        'm_nu': m_nu,
        'w0': w0,
        'wa': wa,
        'tcmb': tcmb,
        'z_vec': np.array([z_vec, 2.1]),
        'k_vec': k_vec,
        'Mmin': mmin,
        'Mmax': mmax,
        'dewiggle': dewiggle,
        'pointmass': pointmass,
        'mdef_model': mdef_model,
        'hmf_model': hmf_model,
        'bias_model': bias_model,
        'halo_profile_model_dm': halo_profile_model_dm,
        'halo_profile_model_sat': halo_profile_model_sat,
        'halo_concentration_model_dm': halo_concentration_model_dm,
        'halo_concentration_model_sat': halo_concentration_model_sat,
        'hmcode_ingredients': hmcode_ingredients,
        'norm_cen': norm_cen,
        'norm_sat': norm_sat,
        'eta_cen': eta_cen,
        'eta_sat': eta_sat,
        'delta_c': delta_c,
        'overdensity': overdensity,
        'log10T_AGN': log10T_AGN,
        'mb': mb,
        't_eff': t_eff,
        'nonlinear_mode': nonlinear_mode,
        'compute_observable': compute_observable,
        'obs_settings': obs_settings,
        'hod_settings': hod_settings,
        'hod_params': hod_params,
        'hod_model': hod_model,
        'rpmin': rpmin,
        'rpmax': rpmax,
        'thetamin': thetamin,
        'thetamax': thetamax,
    }
    current_hash = hash_params(params)
    # --------------------------
    # Compute Current Model
    # --------------------------
    should_run = run_model or not st.session_state.has_run

    params_changed = current_hash != st.session_state.params_hash

    if should_run and params_changed and not no_selection:
        with st.spinner(
            'Calculating the new model, please wait a moment...',
            show_time=True,
        ):
            st.session_state.computed_outputs = compute_outputs(params)

        st.session_state.params_hash = current_hash
        st.session_state.has_run = True

    computed_outputs = st.session_state.get('computed_outputs', None)
    if computed_outputs is not None:
        col1, col2 = st.columns(2, width=500)
        # --------------------------
        # Save Model Button
        # --------------------------
        if col1.button('Add current model for comparison', width='stretch'):
            st.session_state.models.append(
                {'params': params.copy(), 'outputs': computed_outputs.copy()}
            )
        if col1.button('Clear saved models', width='stretch'):
            st.session_state.models = []

        if col2.button('Set current model as reference', width='stretch'):
            st.session_state.reference_model = {'outputs': computed_outputs.copy()}

        if col2.button('Clear reference model', width='stretch'):
            st.session_state.reference_model = None

        selected_outputs = st.session_state.selected_outputs
        if not no_selection:
            tabs = st.tabs(selected_outputs, width='stretch')
            for tab, output in zip(tabs, selected_outputs):
                with tab:
                    category, subtype = OBSERVABLE_MAP[output]
                    if subtype in [
                        'wtheta',
                        'gamma',
                        'xip',
                        'xim',
                    ]:
                        if params['z_vec'][0] == 0.0:
                            st.warning(
                                f'The {output} function is not very well defined at redshift $z = 0$, select a higher redshift and re-run the model! Moreover, it is evaluated at a single redshift with highly simplified projection, thus can only serve as an illustrative example!',
                                icon='⚠️',
                                width=700,
                            )
                        else:
                            st.warning(
                                f'The {output} function is evaluated at a single redshift with highly simplified projection, thus can only serve as an illustrative example!',
                                icon='⚠️',
                                width=700,
                            )

                    if subtype in computed_outputs:
                        x, y = computed_outputs[subtype]

                        fig_main = plot_observable(
                            x, y, subtype, compare_reference, components=components
                        )
                        st.plotly_chart(
                            fig_main,
                            width='content',
                            height='content',
                            key=f'fig_{output}',
                        )
                        if (
                            compare_reference
                            and st.session_state.reference_model is not None
                        ):
                            x_ref, y_ref = st.session_state.reference_model['outputs'][
                                subtype
                            ]
                            fig_ratio = plot_ratio(x, y, x_ref, y_ref, subtype)
                            st.plotly_chart(
                                fig_ratio,
                                width='content',
                                height='content',
                                key=f'fig_{output}_ref',
                            )

                        # CSV download
                        # Create an in-memory buffer
                        with io.BytesIO() as buffer:
                            # Write array to buffer
                            np.savetxt(
                                buffer,
                                np.column_stack((x, y['tot'])),
                                delimiter=',',
                                header='x, y axis',
                            )
                            st.download_button(
                                label='Download data as CSV',
                                data=buffer,
                                file_name=f'{subtype}.csv',
                                mime='text/csv',
                            )
