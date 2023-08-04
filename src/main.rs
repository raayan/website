mod plot;

use std::cell::RefCell;

use nalgebra::{dvector, DVector};
use num_traits::FloatConst;
use once_cell::sync::Lazy;
use plot::gaussian::{BoundedEstimator, MixtureModel};
use plotters::{
    prelude::{ChartBuilder, IntoDrawingArea, IntoLinspace, PathElement, SeriesLabelPosition},
    series::SurfaceSeries,
    style::{
        full_palette::PURPLE, Color, IntoFont, RGBColor, BLACK, BLUE, CYAN, GREEN, MAGENTA, RED,
        WHITE,
    },
};
use plotters_canvas::CanvasBackend;
use rand::prelude::{Distribution, SliceRandom, StdRng};
use rand::{Rng, SeedableRng};
use statrs::distribution::{Continuous, MultivariateNormal, Uniform};
use wasm_bindgen::{prelude::Closure, JsCast, JsValue};
use web_sys::window;

const QUADRANT_SCALE: f64 = 1.0;
const QUADRANT_PAD: f64 = 0.2;
const POINTS: usize = 10;
static BOUNDS: Lazy<DVector<f64>> = Lazy::new(|| dvector![-1.0, 1.0]);

fn main() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let win = window().expect("could not window");

    let document = win.document().expect("could not document");

    let canvas = document
        .get_element_by_id("canvas")
        .expect("expected canvas")
        .dyn_into::<web_sys::HtmlCanvasElement>()?;

    let cs = {
        let root = CanvasBackend::with_canvas_object(canvas.clone())
            .unwrap()
            .into_drawing_area();

        root.fill(&WHITE).map_err(|s| s.to_string())?;

        let quadrant_size = QUADRANT_SCALE + QUADRANT_PAD;
        let x_axis = (-quadrant_size..quadrant_size).step(0.1);
        let z_axis = (-quadrant_size..quadrant_size).step(0.1);
        let y_axis = (0.0..quadrant_size).step(0.1);

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .set_all_label_area_size(30)
            .caption("2D MV-Normal", ("monospace", 20))
            .build_cartesian_3d(x_axis, y_axis, z_axis)
            .map_err(|s| s.to_string())?;

        chart
            .plotting_area()
            .fill(&WHITE)
            .map_err(|s| s.to_string())?;

        chart.with_projection(|mut pb| {
            pb.pitch = f64::PI() * (50. / 360.);
            pb.yaw = f64::PI() * (120. / 360.);
            pb.scale = 0.8;
            pb.into_matrix()
        });

        chart
            .configure_axes()
            .label_style(("monospace", 15).into_font().color(&BLACK))
            .light_grid_style(BLACK.mix(0.20))
            .max_light_lines(10)
            .draw()
            .map_err(|s| s.to_string())?;

        let cs = chart.into_chart_state();
        cs
    };

    let mixture_rc: RefCell<BoundedEstimator<MultivariateNormal, _, _>> =
        RefCell::new(BoundedEstimator::new((&BOUNDS, &BOUNDS)));
    let sub_colors_rc: RefCell<Vec<RGBColor>> = RefCell::new(Vec::new());

    let colors = vec![BLUE, RED, GREEN, MAGENTA, CYAN, PURPLE];

    let a = Closure::<dyn Fn()>::new(move || {
        let count: usize = canvas
            .get_attribute("plot-yaw")
            .expect("no yaw")
            .parse()
            .expect("could not parse yaw");

        let root = CanvasBackend::with_canvas_object(canvas.clone())
            .unwrap()
            .into_drawing_area();

        let mut chart = cs.clone().restore(&root);
        chart.plotting_area().fill(&WHITE).unwrap();

        chart
            .configure_axes()
            .label_style(("monospace", 15).into_font().color(&BLACK))
            .light_grid_style(BLACK.mix(0.20))
            .max_light_lines(10)
            .draw()
            .unwrap();

        let uniform = Uniform::new(BOUNDS[0], BOUNDS[1]).unwrap();
        let mut mixture = mixture_rc.borrow_mut();
        let mut sub_colors = sub_colors_rc.borrow_mut();
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();

        let base = (0_u64..).take(POINTS);
        let elems = base.clone().count();
        let points = base
            .clone()
            .map(|i| -QUADRANT_SCALE + ((2.0 * QUADRANT_SCALE) * (i as f64) / (elems as f64)));

        let observations = mixture.observations.len();
        for (dist, color) in mixture
            .observations
            .iter()
            .rev()
            .take(10)
            .zip(sub_colors.iter())
        {
            chart
                .draw_series(
                    SurfaceSeries::xoz(points.clone(), points.clone(), |x, y| {
                        dist.pdf(&dvector![x, y]) / (observations as f64)
                    })
                    .style(color.mix(0.2)),
                )
                .unwrap();
        }

        chart
            .draw_series(
                SurfaceSeries::xoz(points.clone(), points.clone(), |x, y| {
                    mixture.pdf(&dvector![x, y])
                })
                .style(&BLACK.mix(0.5)),
            )
            .unwrap()
            .label(format!("{:06} observations", observations))
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));

        chart
            .configure_series_labels()
            .label_font(("monospace", 15).into_font().color(&BLACK))
            .border_style(BLACK)
            .position(SeriesLabelPosition::UpperLeft)
            .draw()
            .unwrap();

        mixture.observe(
            MultivariateNormal::new(
                vec![uniform.sample(&mut rng), uniform.sample(&mut rng)],
                vec![
                    rng.gen_range(0.0001..0.1),
                    0.0,
                    0.0,
                    rng.gen_range(0.0001..0.1),
                ],
            )
            .unwrap(),
        );
        sub_colors.push(*colors.choose(&mut rng).unwrap());

        root.present().unwrap();

        canvas
            .set_attribute("plot-yaw", &(count + 1).to_string())
            .expect("could not set yaw");
    });

    win.set_interval_with_callback_and_timeout_and_arguments_0(a.as_ref().unchecked_ref(), 100)?;

    a.forget();

    Ok(())
}
