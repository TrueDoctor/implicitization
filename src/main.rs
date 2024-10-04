use std::collections::LinkedList;

use approx::{assert_relative_eq, assert_relative_ne};
use glam::{DVec2, Vec2Swizzles};
use roots::{find_root_newton_raphson, find_roots_cubic};

#[derive(Clone, Debug)]
struct CubicBezier {
    control_points: [DVec2; 4],
    // Precomputed values for efficient evaluation
    a: DVec2,
    b: DVec2,
    c: DVec2,
    d: DVec2,
}

#[derive(Clone, Debug)]
struct ImplicitCubic {
    coefficients: [f64; 4],
    control_points: [DVec2; 4],
    // Precomputed values for efficient derivative evaluation
    dl_dx: [f64; 10],
    dl_dy: [f64; 10],
}

struct BernsteinBezier {
    control_points: [DVec2; 4],
    weights: [f64; 4],
}

impl CubicBezier {
    fn new(control_points: [DVec2; 4]) -> Self {
        let [p0, p1, p2, p3] = control_points;
        let a = -p0 + 3.0 * p1 - 3.0 * p2 + p3;
        let b = 3.0 * p0 - 6.0 * p1 + 3.0 * p2;
        let c = -3.0 * p0 + 3.0 * p1;
        let d = p0;

        CubicBezier {
            control_points,
            a,
            b,
            c,
            d,
        }
    }

    fn evaluate(&self, t: f64) -> DVec2 {
        self.a * t.powi(3) + self.b * t.powi(2) + self.c * t + self.d
    }

    fn derivative(&self, t: f64) -> DVec2 {
        3.0 * self.a * t.powi(2) + 2.0 * self.b * t + self.c
    }

    fn second_derivative(&self, t: f64) -> DVec2 {
        6.0 * self.a * t + 2.0 * self.b
    }

    fn point_to_parameter(&self, point: DVec2) -> Option<f64> {
        // Define a cubic equation solver
        fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
            let roots = find_roots_cubic(a, b, c, d);
            dbg!(&roots);
            roots
                .as_ref()
                .iter()
                .filter(|&&r| (0.0..=1.0).contains(&r))
                .copied()
                .collect()
        }

        let [p0, p1, p2, p3] = self.control_points;
        let ax = -p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x;
        let bx = 3.0 * p0.x - 6.0 * p1.x + 3.0 * p2.x;
        let cx = -3.0 * p0.x + 3.0 * p1.x;
        let dx = p0.x - point.x;

        let ay = -p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y;
        let by = 3.0 * p0.y - 6.0 * p1.y + 3.0 * p2.y;
        let cy = -3.0 * p0.y + 3.0 * p1.y;
        let dy = p0.y - point.y;

        let tx = solve_cubic(ax, bx, cx, dx);
        let ty = solve_cubic(ay, by, cy, dy);

        // Find a common root
        dbg!(&tx, &ty);
        tx.into_iter()
            .find(|&t1| ty.iter().any(|&t2| (t1 - t2).abs() < 1e-8))
    }

    fn point_to_parameter_numerical(&self, point: DVec2, epsilon: f64) -> Option<f64> {
        // First, perform a coarse search using binary search
        let mut t_min = 0.0;
        let mut t_max = 1.0;
        let mut best_t = 0.5;
        let mut best_distance = f64::INFINITY;

        for i in 0..200 {
            let t = (i as f64) / 10.;
            let p = self.evaluate(t);

            let d = p.distance_squared(point);

            if d < best_distance {
                best_t = t;
                best_distance = d;
            }
        }

        // Refine the result using Newton-Raphson method
        let mut t = best_t;
        for _ in 0..100 {
            // Adjust the number of iterations as needed
            let p = self.evaluate(t);
            let dp = self.derivative(t);

            let f = p - point;
            let df = dp;

            let delta_t = (f.dot(df)) / (df.dot(df) + epsilon);
            t -= delta_t;

            t = t.clamp(0.0, 1.0);

            if delta_t.abs() < epsilon {
                break;
            }
        }

        // Check if the point is actually on the curve
        let final_point = self.evaluate(t);
        if dbg!(final_point.distance(point)) < epsilon {
            Some(t)
        } else {
            None
        }
    }

    fn sample_at(&self, t: f64) -> DVec2 {
        let [p1, p2, p3, p4] = self.control_points;
        let p01 = p1.lerp(p2, t);
        let p12 = p2.lerp(p3, t);
        let p23 = p3.lerp(p4, t);
        let p012 = p01.lerp(p12, t);
        let p123 = p12.lerp(p23, t);
        p012.lerp(p123, t)
    }
    fn to_implicit(&self) -> Option<ImplicitCubic> {
        let lambda = self.calculate_lambda()?;
        let phi = self.calculate_phi(&lambda);
        let coefficients = self.calculate_coefficients(&lambda, &phi);
        Some(ImplicitCubic::new(coefficients, self.control_points))
    }

    fn calculate_lambda(&self) -> Option<[f64; 4]> {
        let lambda = |i: usize, j: usize, k: usize| -> Option<f64> {
            let ci = self.control_points[i];
            let cj = self.control_points[j];
            let ck = self.control_points[k];
            let positive = ci.x * cj.y + ci.y * ck.x + cj.x * ck.y;
            let negative = ck.x * cj.y + ck.y * ci.x + cj.x * ci.y;
            let result = positive - negative;
            let result = (cj.x - ci.x) * (ck.y - ci.y) - (ck.x - ci.x) * (cj.y - ci.y);
            // eprintln!("lambda({}, {}, {}) = {}", i, j, k, result);
            // TODO: subdivide
            // assert_relative_ne!(result, 0., epsilon = 1e-6);
            if result.abs() < 1e-8 {
                // println!("colinear case");
                return None;
            }
            Some(result)
        };

        let result = [
            lambda(3, 2, 1)?,
            lambda(2, 3, 0)?,
            lambda(1, 0, 3)?,
            lambda(0, 1, 2)?,
        ];
        assert_relative_eq!(result.iter().sum::<f64>(), 0., epsilon = 1e-8);
        // eprintln!("Final lambda: {:?}", result);
        Some(result)
    }

    fn calculate_phi(&self, lambda: &[f64; 4]) -> [f64; 3] {
        let u = [1.0, 3.0, 3.0, 1.0];
        let result = [
            u[0] * u[2] * lambda[1].powi(2) - u[1].powi(2) * lambda[0] * lambda[2],
            u[1] * u[3] * lambda[2].powi(2) - u[2].powi(2) * lambda[1] * lambda[3],
            u[1] * u[2] * lambda[0] * lambda[3] - u[0] * u[3] * lambda[1] * lambda[2],
        ];
        // eprintln!("Phi calculation: {:?}", result);
        result
    }

    fn calculate_coefficients(&self, lambda: &[f64; 4], phi: &[f64; 3]) -> [f64; 4] {
        let u = [1.0, 3.0, 3.0, 1.0];

        let b_tilde = [
            u[1] * u[2] / (u[0] * u[3]) - lambda[1] * lambda[2] / (lambda[0] * lambda[3]),
            lambda[1].powi(2) / (lambda[0] * lambda[2]) - u[1].powi(2) / (u[0] * u[2]),
            lambda[2].powi(2) / (lambda[1] * lambda[3]) - u[2].powi(2) / (u[1] * u[3]),
            lambda[0] * lambda[3] / (lambda[1] * lambda[2]) - u[0] * u[3] / (u[1] * u[2]),
        ];

        // Normalize the coefficients
        let norm_factor = b_tilde.iter().map(|&x| x.abs()).sum::<f64>().recip();
        let result = b_tilde.map(|x| x * norm_factor);
        // let result = [
        //     phi[2] * u[1] * u[2] * lambda[1] * lambda[2],
        //     phi[0] * u[1] * u[3] * lambda[1] * lambda[3],
        //     phi[1] * u[0] * u[2] * lambda[0] * lambda[2],
        //     phi[2] * u[0] * u[3] * lambda[0] * lambda[3],
        // ];
        // eprintln!("Coefficients calculation: {:?}", result);
        result
    }
    fn subdivide(&self, t: f64) -> (CubicBezier, CubicBezier) {
        let [p0, p1, p2, p3] = self.control_points;
        let q0 = p0.lerp(p1, t);
        let q1 = p1.lerp(p2, t);
        let q2 = p2.lerp(p3, t);
        let r0 = q0.lerp(q1, t);
        let r1 = q1.lerp(q2, t);
        let s = r0.lerp(r1, t);

        (
            CubicBezier::new([p0, q0, r0, s]),
            CubicBezier::new([s, r1, q2, p3]),
        )
    }
}

impl ImplicitCubic {
    fn new(coefficients: [f64; 4], control_points: [DVec2; 4]) -> Self {
        let mut dl_dx = [0.0; 10];
        let mut dl_dy = [0.0; 10];

        for i in 0..3 {
            for j in (i + 1)..4 {
                let idx = i * 3 + j - 1;
                dl_dx[idx] = control_points[j].y - control_points[i].y;
                dl_dy[idx] = control_points[i].x - control_points[j].x;
            }
        }

        ImplicitCubic {
            coefficients,
            control_points,
            dl_dx,
            dl_dy,
        }
    }

    fn evaluate(&self, point: DVec2) -> f64 {
        let l = |i: usize, j: usize| -> f64 {
            let c_i = self.control_points[i];
            let c_j = self.control_points[j];
            let result = (point.x - c_i.x) * (c_j.y - c_i.y) - (point.y - c_i.y) * (c_j.x - c_i.x);
            // eprintln!("L{}{}({:?}) = {}", i, j, point, result);
            result
        };

        let k0 = l(0, 1) * l(1, 2) * l(2, 3);
        let k1 = l(0, 1) * l(1, 3).powi(2);
        let k2 = l(0, 2).powi(2) * l(2, 3);
        let k3 = l(0, 3).powi(3);

        self.coefficients[0] * k0
            + self.coefficients[1] * k1
            + self.coefficients[2] * k2
            + self.coefficients[3] * k3
    }

    fn gradient(&self, point: DVec2) -> DVec2 {
        let l = |i: usize, j: usize| -> f64 {
            let c_i = self.control_points[i];
            let c_j = self.control_points[j];
            let result = (point.x - c_i.x) * (c_j.y - c_i.y) - (point.y - c_i.y) * (c_j.x - c_i.x);
            // eprintln!("L{}{}({:?}) = {}", i, j, point, result);
            result
        };

        let dl = |i: usize, j: usize| -> DVec2 {
            let c_i = self.control_points[i];
            let c_j = self.control_points[j];
            DVec2::new(c_j.y - c_i.y, c_i.x - c_j.x)
        };

        let dk0 = dl(0, 1) * l(1, 2) * l(2, 3)
            + l(0, 1) * dl(1, 2) * l(2, 3)
            + l(0, 1) * l(1, 2) * dl(2, 3);
        let dk1 = dl(0, 1) * l(1, 3).powi(2) + 2.0 * l(0, 1) * l(1, 3) * dl(1, 3);
        let dk2 = 2.0 * l(0, 2) * dl(0, 2) * l(2, 3) + l(0, 2).powi(2) * dl(2, 3);
        let dk3 = 3.0 * l(0, 3).powi(2) * dl(0, 3);

        self.coefficients[0] * dk0
            + self.coefficients[1] * dk1
            + self.coefficients[2] * dk2
            + self.coefficients[3] * dk3
    }

    // fn evaluate(&self, point: DVec2) -> f64 {
    //     let l = |i: usize, j: usize| -> f64 {
    //         let c_i = self.control_points[i];
    //         let c_j = self.control_points[j];
    //         let result = (point.x - c_i.x) * (c_j.y - c_i.y) - (point.y - c_i.y) * (c_j.x - c_i.x);
    //         // eprintln!("L{}{}({:?}) = {}", i, j, point, result);
    //         result
    //     };

    //     let k0 = l(0, 1) * l(1, 2) * l(2, 3);
    //     let k1 = l(0, 1) * l(1, 3).powi(2);
    //     let k2 = l(0, 2).powi(2) * l(2, 3);
    //     let k3 = l(0, 3).powi(3);

    //     // eprintln!(
    //     //     "K values: k0 = {}, k1 = {}, k2 = {}, k3 = {}",
    //     //     k0, k1, k2, k3
    //     // );

    //     let result = self.coefficients[0] * k0
    //         + self.coefficients[1] * k1
    //         + self.coefficients[2] * k2
    //         + self.coefficients[3] * k3;

    //     // eprintln!("Evaluation result: {}", result);
    //     result
    // }
}

fn newton_raphson_2d(
    mut guess: DVec2,
    system: impl Fn(DVec2) -> [f64; 2],
    jacobian: impl Fn(DVec2) -> [[f64; 2]; 2],
    max_iterations: usize,
    tolerance: f64,
) -> Option<DVec2> {
    for _ in 0..max_iterations {
        let f = system(guess);
        let j = jacobian(guess);

        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        if det.abs() < 1e-10 {
            return None; // Jacobian is singular
        }

        let dx = (j[1][1] * f[0] - j[0][1] * f[1]) / det;
        let dy = (-j[1][0] * f[0] + j[0][0] * f[1]) / det;

        guess -= DVec2::new(dx, dy);

        if dx.abs() < tolerance && dy.abs() < tolerance {
            return Some(guess);
        }
    }
    None
}

// fn find_intersections(curve1: &CubicBezier, curve2: &CubicBezier) -> Option<Vec<(DVec2, DVec2)>> {
//     let implicit1 = curve1.to_implicit()?;
//     let implicit2 = curve2.to_implicit()?;

//     let system =
//         |point: DVec2| -> [f64; 2] { [implicit1.evaluate(point), implicit2.evaluate(point)] };

//     let jacobian = |point: DVec2| -> [[f64; 2]; 2] {
//         let h = 1e-6;
//         let fx = system(point + DVec2::new(h, 0.0));
//         let fy = system(point + DVec2::new(0.0, h));
//         let f = system(point);
//         [
//             [(fx[0] - f[0]) / h, (fy[0] - f[0]) / h],
//             [(fx[1] - f[1]) / h, (fy[1] - f[1]) / h],
//         ]
//     };

//     let initial_guesses = generate_initial_guesses(curve1, curve2);

//     let mut intersection_points: Vec<_> = initial_guesses
//         .iter()
//         .flat_map(|guess| newton_raphson_2d(*guess, system, jacobian, 100, 1e-8))
//         .collect();
//     intersection_points.sort_unstable_by(|a, b| {
//         (a.x, a.y)
//             .partial_cmp(&(b.x, b.y))
//             .unwrap_or(std::cmp::Ordering::Equal)
//     });
//     intersection_points.dedup_by_key(|point| (*point * 1e8).as_i64vec2());
//     // dbg!(&intersection_points);
//     let mut intersections = Vec::new();
//     for point in intersection_points {
//         if let (Some(t1), Some(t2)) = (
//             curve1.point_to_parameter(point),
//             curve2.point_to_parameter(point),
//         ) {
//             if t1.abs() < 1e-12 || t2.abs() < 1e-12 {
//                 continue;
//             }
//             if (t1 - 1.).abs() < 1e-12 || (t2 - 1.).abs() < 1e-12 {
//                 continue;
//             }
//             intersections.push((point, (t1, t2).into()));
//         }
//     }

//     Some(intersections)
// }

fn find_intersections(curve1: &CubicBezier, curve2: &CubicBezier) -> Option<Vec<(DVec2, DVec2)>> {
    fn find_intersections_recursive(
        curve1: &CubicBezier,
        curve2: &CubicBezier,
        min_t: DVec2,
        max_t: DVec2,
        depth: usize,
    ) -> Option<Vec<(DVec2, DVec2)>> {
        if depth > 5 {
            eprintln!("\n\n # SKIP \n \n");
            return None; // Prevent infinite recursion
        }

        let implicit1 = curve1.to_implicit();
        let implicit2 = curve2.to_implicit();

        match (implicit1, implicit2) {
            (/*Some(imp1)*/ _, Some(imp2)) => {
                find_intersection_no_collinear(/*&imp1,*/ &imp2, curve1, curve2, min_t, max_t)
            }
            (Some(imp1), _) => Some(
                find_intersection_no_collinear(&imp1, curve1, curve2, min_t.yx(), max_t.yx())?
                    .into_iter()
                    .map(|(point, st)| (point.yx(), st.yx()))
                    .collect(),
            ),
            (None, _) => {
                // Subdivide curve1
                let t_mid = (min_t.x + max_t.x) / 2.0;
                let (left, right) = curve1.subdivide(0.5);
                let mut intersections = Vec::new();

                let mut left_intersections = find_intersections_recursive(
                    &left,
                    curve2,
                    min_t,
                    DVec2::new(t_mid, max_t.y),
                    depth + 1,
                )?;
                intersections.append(&mut left_intersections);

                let mut right_intersections = find_intersections_recursive(
                    &right,
                    curve2,
                    DVec2::new(t_mid, min_t.y),
                    max_t,
                    depth + 1,
                )?;
                intersections.append(&mut right_intersections);

                Some(intersections)
            }
            (_, _) => Some(
                find_intersections_recursive(curve2, curve1, min_t.yx(), max_t.yx(), depth + 1)?
                    .into_iter()
                    .map(|(point, st)| (point.yx(), st.yx()))
                    .collect(),
            ),
        }
    }

    find_intersections_recursive(curve1, curve2, DVec2::ZERO, DVec2::ONE, 0)
}

// fn find_intersection_no_collinear(
//     imp1: ImplicitCubic,
//     imp2: ImplicitCubic,
//     curve1: &CubicBezier,
//     curve2: &CubicBezier,
//     t1_min: f64,
//     t1_max: f64,
//     t2_min: f64,
//     t2_max: f64,
// ) -> Option<Vec<(DVec2, DVec2)>> {
//     // Proceed with intersection finding as before
//     let system = |point: DVec2| -> [f64; 2] { [imp1.evaluate(point), imp2.evaluate(point)] };
//     let jacobian = |point: DVec2| -> [[f64; 2]; 2] {
//         let h = 1e-6;
//         let fx = system(point + DVec2::new(h, 0.0));
//         let fy = system(point + DVec2::new(0.0, h));
//         let f = system(point);
//         [
//             [(fx[0] - f[0]) / h, (fy[0] - f[0]) / h],
//             [(fx[1] - f[1]) / h, (fy[1] - f[1]) / h],
//         ]
//     };

//     let initial_guesses = generate_initial_guesses(curve1, curve2);
//     let mut intersection_points: Vec<_> = initial_guesses
//         .iter()
//         .flat_map(|guess| newton_raphson_2d(*guess, system, jacobian, 100, 1e-8))
//         .collect();
//     intersection_points.sort_unstable_by(|a, b| {
//         (a.x, a.y)
//             .partial_cmp(&(b.x, b.y))
//             .unwrap_or(std::cmp::Ordering::Equal)
//     });
//     intersection_points.dedup_by_key(|point| (*point * 1e8).as_i64vec2());

//     // Map the t values to the correct range
//     let intersections: Vec<(DVec2, DVec2)> = intersection_points
//         .into_iter()
//         .filter_map(|point| {
//             let t1 = curve1.point_to_parameter(point)?;
//             let t2 = curve2.point_to_parameter(point)?;
//             if t1.abs() < 1e-12 || t2.abs() < 1e-12 {
//                 return None;
//             }
//             if (t1 - 1.).abs() < 1e-12 || (t2 - 1.).abs() < 1e-12 {
//                 return None;
//             }
//             let mapped_t1 = t1_min + (t1_max - t1_min) * t1;
//             let mapped_t2 = t2_min + (t2_max - t2_min) * t2;
//             Some((point, DVec2::new(mapped_t1, mapped_t2)))
//         })
//         .collect();

//     Some(intersections)
// }

fn find_intersection_no_collinear(
    // imp1: &ImplicitCubic,
    imp2: &ImplicitCubic,
    curve1: &CubicBezier,
    curve2: &CubicBezier,
    min_t: DVec2,
    max_t: DVec2,
) -> Option<Vec<(DVec2, DVec2)>> {
    let mut intersections = Vec::new();
    let mut t = 0.;
    let mut step = 0.01; // Initial step size, can be adjusted

    let mut last_f2 = imp2.evaluate(curve1.evaluate(t));
    let mut last_df2 = imp2.gradient(curve1.evaluate(t)).dot(curve1.derivative(t));
    let map_t = |x, y| min_t + (max_t - min_t) * DVec2::new(x, y);

    while t < 1. {
        let point = curve1.evaluate(t);
        let f2 = imp2.evaluate(point);
        let df2 = imp2.gradient(point).dot(curve1.derivative(t));

        if f2.signum() != last_f2.signum() {
            eprintln!("refining interrsection:");
            dbg!(t, f2, df2);
            // We've crossed an intersection, use Newton's to refine it
            if let Some((refined_t, refined_point)) =
                newton_refine_intersection(curve1, imp2, t - step, t)
            {
                dbg!(refined_t, refined_point);
                if let Some(t2) = dbg!(curve2.point_to_parameter_numerical(refined_point, 1e-7)) {
                    // if let Some(t2) = dbg!(curve2.point_to_parameter(refined_point)) {
                    intersections.push((refined_point, map_t(refined_t, t2)));
                }
            }

            // The next point will be a min/max, so we can skip ahead
            t += step;
            last_f2 = imp2.evaluate(curve1.evaluate(t));
            last_df2 = imp2.gradient(curve1.evaluate(t)).dot(curve1.derivative(t));
            continue;
        }

        if df2.signum() != last_df2.signum() {
            // We've passed a local min/max
            if let Some((refined_t, refined_point)) =
                newton_refine_extremum(curve1, imp2, t - step, t)
            {
                // Check if this extremum is actually an intersection
                let extremum_f2 = imp2.evaluate(refined_point);
                if extremum_f2.abs() < 1e-8 {
                    if let Some(t2) = curve2.point_to_parameter(refined_point) {
                        intersections.push((refined_point, map_t(refined_t, t2)));
                    }
                }
            }

            // Move just past the extremum
            t += step;
        } else {
            // No intersection or extremum found, continue along curve
            t += step;
        }

        // Adaptive step size
        step = if df2.abs() > 1e-8 {
            (0.01 * f2.abs() / df2.abs()).min(0.1).max(0.001)
        } else {
            0.01
        };

        last_f2 = f2;
        last_df2 = df2;
    }

    Some(intersections)
}

fn newton_refine_intersection(
    curve: &CubicBezier,
    implicit: &ImplicitCubic,
    mut t_start: f64,
    mut t_end: f64,
) -> Option<(f64, DVec2)> {
    dbg!(t_start, t_end);
    let mut t = (t_start + t_end) / 2.0;
    for _ in 0..100 {
        // Max iterations
        // let t = (t_start + t_end) / 2.0;
        let point = curve.evaluate(t);
        let f = implicit.evaluate(point);
        let df = implicit.gradient(point).dot(curve.derivative(t));
        // dbg!(t, f, df);

        if f.abs() < 1e-10 {
            // eprintln!("found intersection {t}");
            return Some((t, point));
        }

        if df.abs() < 1e-8 {
            eprintln!("found stationary point");
            return None; // Stationary point, not an intersection
        }

        t -= f / df;

        if t < t_start - 0.1 || t > t_end + 0.1 {
            eprintln!("leaving interval {t}");
            return None; // New t is outside the interval
        }

        // if t_new > t {
        //     t_end = t;
        // } else {
        //     t_start = t;
        // }
    }
    eprintln!("did not converge ");
    None // Did not converge
}

fn newton_refine_extremum(
    curve: &CubicBezier,
    implicit: &ImplicitCubic,
    mut t_start: f64,
    mut t_end: f64,
) -> Option<(f64, DVec2)> {
    for _ in 0..10 {
        // Max iterations
        let t = (t_start + t_end) / 2.0;
        let point = curve.evaluate(t);
        let df = implicit.gradient(point).dot(curve.derivative(t));
        let d2f = implicit.gradient(point).dot(curve.second_derivative(t))
            + curve.derivative(t).dot(jacobian_vector_product(
                implicit,
                point,
                curve.derivative(t),
            ));

        if df.abs() < 1e-8 {
            return Some((t, point));
        }

        if d2f.abs() < 1e-8 {
            return None; // Inflection point or degenerate case
        }

        let t_new = t - df / d2f;

        if t_new < t_start || t_new > t_end {
            return None; // New t is outside the interval
        }

        if df > 0.0 {
            t_end = t;
        } else {
            t_start = t;
        }
    }
    None // Did not converge
}

fn jacobian_vector_product(implicit: &ImplicitCubic, point: DVec2, vector: DVec2) -> DVec2 {
    let epsilon = 1e-8;
    let f_plus = implicit.gradient(point + epsilon * vector);
    let f_minus = implicit.gradient(point - epsilon * vector);
    (f_plus - f_minus) / (2.0 * epsilon)
}

fn generate_initial_guesses(curve1: &CubicBezier, curve2: &CubicBezier) -> Vec<DVec2> {
    let mut guesses = Vec::new();
    for i in 0..10 {
        guesses.push(curve1.sample_at(100. / i as f64));
    }
    for i in 0..10 {
        guesses.push(curve1.sample_at(1. - (100. / i as f64)));
    }
    for &p1 in &curve1.control_points {
        for &p2 in &curve2.control_points {
            guesses.push((p1 + p2) * 0.5);
        }
    }
    guesses
}

fn compute_bounding_box(curves: &[&CubicBezier]) -> (DVec2, DVec2) {
    let mut min = DVec2::splat(f64::INFINITY);
    let mut max = DVec2::splat(f64::NEG_INFINITY);

    for curve in curves {
        for &point in &curve.control_points {
            min = min.min(point);
            max = max.max(point);
        }
    }

    (min, max)
}

fn normalize_curve(curve: &CubicBezier, min: DVec2, max: DVec2) -> CubicBezier {
    let scale = 10. / (max - min);
    let mid = (max + min) * 0.5;
    CubicBezier::new(curve.control_points.map(|p| (p - mid) * scale))
}

fn main() {
    // Example usage
    let curve1 = CubicBezier::new([
        // DVec2::new(0.0, 0.0),
        // DVec2::new(1.0, 2.0),
        // DVec2::new(3.0, 3.0),
        // DVec2::new(4.0, 1.0),
        // DVec2::new(24.0, 23.947789),
        // DVec2::new(11.685371, 23.947789),
        // DVec2::new(3.3410435, 7.9372144),
        // DVec2::new(43.299408, 7.9372144),
        // DVec2::new(-1.6547964731782374, -5.000000000000001),
        // DVec2::new(-1.9444534046484843, -4.094338068015234),
        // DVec2::new(-3.103788215240571, -0.05473070972497621),
        // DVec2::new(-3.3936808416145166, 0.7876948017272548),
        // DVec2::new(-5.000000000000003, 4.934462998421441),
        // DVec2::new(-2.92543314862846, 0.38747658364775717),
        // DVec2::new(-1.8466881660765833, -4.6891194991001965),
        // DVec2::new(-1.8247157521781525, -4.793510086734055),
        DVec2::new(-2.1329297518905785, -4.660725633196663),
        DVec2::new(0.3427083767744279, -2.9606197198656288),
        DVec2::new(4.117192929548024, 0.08071496516017329),
        DVec2::new(4.999999999999998, 4.901871928855905),
    ]);

    let curve2 = CubicBezier::new([
        // DVec2::new(0.0, 1.0),
        // DVec2::new(2.0, 3.0),
        // DVec2::new(3.0, 2.0),
        // DVec2::new(4.0, 0.0),
        // DVec2::new(3.2052288, 7.1229882),
        // DVec2::new(42.417605, 7.1229882),
        // DVec2::new(36.314629, 23.947789),
        // DVec2::new(24.0, 23.947789),
        // DVec2::new(-5.000000000000001, -2.838848069852117),
        // DVec2::new(-0.19244830680270536, -0.201395232460811),
        // DVec2::new(0.11694561621364338, 3.284920308298407),
        // DVec2::new(4.999999999999998, 5.0),
        // DVec2::new(4.999999999999998, 4.9999999999999964),
        // DVec2::new(4.225801709941227, -0.008537541579500629),
        // DVec2::new(0.39817602875185254, -3.2264595857075933),
        // DVec2::new(-2.1217092285411083, -5.0000000000000036),
        DVec2::new(-5.000000000000003, 5.0),
        DVec2::new(-2.8472549665952513, 0.32586779421415046),
        DVec2::new(-1.7278561069764025, -4.892688102167379),
        DVec2::new(-1.7050554973219245, -5.0),
    ]);

    println!("Curve 1: {:?}", curve1);
    println!("Curve 2: {:?}", curve2);
    let (min, max) = compute_bounding_box(&[&curve1, &curve2]);
    let curve1 = normalize_curve(&curve1, min, max);
    let curve2 = normalize_curve(&curve2, min, max);
    println!("normalized Curve 1: {:?}", curve1);
    println!("normalized Curve 2: {:?}", curve2);

    // Convert to implicit form
    let implicit1 = curve1.to_implicit().unwrap();
    let implicit2 = curve2.to_implicit().unwrap();

    // let intersection = curve1.sample_at(0.5);
    // let intersection = curve1.sample_at(0.7956691588529524);
    // let intersection = curve1.sample_at(0.6050790720042927);
    // let intersection = curve1.sample_at(0.9903600370618053);
    let intersection = curve1.sample_at(0.039966561782367335);

    println!("Implicit form of Curve 1: {:?}", implicit1);
    println!("Implicit form of Curve 2: {:?}", implicit2);

    println!("eval 1: {}", implicit1.evaluate(intersection));
    println!("eval 2: {}", implicit2.evaluate(intersection));

    // Find intersections
    let intersections = find_intersections(&curve1, &curve2).unwrap();

    println!("Intersections found: {:?}", intersections);

    // Verify intersections
    for (i, (point, st)) in intersections.iter().enumerate() {
        let eval1 = implicit1.evaluate(*point);
        let eval2 = implicit2.evaluate(*point);
        println!("Intersection {}: {:?}", i + 1, point);
        println!("  Parameter on Curve 1: {}", st.x);
        println!("  Parameter on Curve 2: {}", st.y);
        println!("  Evaluation on Curve 1: {}", eval1);
        println!("  Evaluation on Curve 2: {}", eval2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_cubic_bezier(points: [(f64, f64); 4]) -> CubicBezier {
        CubicBezier::new(points.map(|(x, y)| DVec2::new(x, y)))
    }

    #[test]
    fn test_example_1() {
        let curve = create_cubic_bezier([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]);
        eprintln!("Control points: {:?}", curve.control_points);

        // Test lambda calculation
        let lambda = curve.calculate_lambda().unwrap();
        assert_relative_eq!(lambda[..], [1.0, -1.0, 1.0, -1.0][..], epsilon = 1e-10);

        // Test phi calculation
        let phi = curve.calculate_phi(&lambda);
        assert_relative_eq!(phi[..], [-6.0, -6.0, -8.0][..], epsilon = 1e-10);

        // Test coefficient calculation
        let coefficients = curve.calculate_coefficients(&lambda, &phi);
        assert_relative_eq!(
            coefficients[..],
            [72.0, -18.0, -18.0, 8.0][..],
            epsilon = 1e-10
        );

        // Test implicit form
        let implicit = curve.to_implicit().unwrap();
        assert_relative_eq!(
            implicit.coefficients[..],
            [72.0, -18.0, -18.0, 8.0][..],
            epsilon = 1e-10
        );

        // Test evaluation of implicit form
        assert_relative_eq!(
            implicit.evaluate(DVec2::new(0.5, 0.75)),
            0.0,
            epsilon = 1e-10
        );

        // Test double point calculation
        // let double_point = calculate_double_point(&curve);
        // assert_relative_eq!(double_point.x, 0.5, epsilon = 1e-10);
        // assert_relative_eq!(double_point.y, -1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_example_2() {
        let curve = create_cubic_bezier([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]);
        eprintln!("Control points: {:?}", curve.control_points);

        let lambda = curve.calculate_lambda().unwrap();
        assert_relative_eq!(lambda[..], [1.0, -1.0, -1.0, 1.0][..], epsilon = 1e-10);

        let phi = curve.calculate_phi(&lambda);
        assert_relative_eq!(phi[..], [12.0, 12.0, 8.0][..], epsilon = 1e-10);

        let coefficients = curve.calculate_coefficients(&lambda, &phi);
        assert_relative_eq!(
            coefficients[..],
            [72.0, -36.0, -36.0, 8.0][..],
            epsilon = 1e-10
        );

        let implicit = curve.to_implicit().unwrap();
        assert_relative_eq!(
            implicit.coefficients[..],
            [72.0, -36.0, -36.0, 8.0][..],
            epsilon = 1e-10
        );

        // Test evaluation of implicit form
        assert_relative_eq!(
            implicit.evaluate(DVec2::new(0.5, 0.5)),
            0.0,
            epsilon = 1e-10
        );

        // let double_point = calculate_double_point(&curve);
        // assert_relative_eq!(double_point.x, 0.5, epsilon = 1e-10);
        // assert_relative_eq!(double_point.y, 0.75, epsilon = 1e-10);
    }

    #[test]
    fn test_example_3() {
        let curve = create_cubic_bezier([(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]);

        let lambda = curve.calculate_lambda().unwrap();
        assert_relative_eq!(lambda[..], [-1.0, 1.0, 1.0, -1.0][..], epsilon = 1e-10);

        let phi = curve.calculate_phi(&lambda);
        assert_relative_eq!(phi[..], [12.0, 12.0, 8.0][..], epsilon = 1e-10);

        let coefficients = curve.calculate_coefficients(&lambda, &phi);
        assert_relative_eq!(
            coefficients[..],
            [72.0, -36.0, -36.0, 8.0][..],
            epsilon = 1e-10
        );

        let implicit = curve.to_implicit().unwrap();
        assert_relative_eq!(
            implicit.coefficients[..],
            [72.0, -36.0, -36.0, 8.0][..],
            epsilon = 1e-10
        );

        // Note: Double point is at infinity for this example
        // We should handle this case in the calculate_double_point function
    }

    // Helper function to calculate double point
    fn calculate_double_point(curve: &CubicBezier) -> DVec2 {
        let lambda = curve.calculate_lambda().unwrap();
        let phi = curve.calculate_phi(&lambda);
        let [c0, c1, c2, c3] = curve.control_points;

        let numerator = -c0 * phi[0].powi(2) * 3.0 * 3.0 + c3 * phi[1].powi(2) * 1.0 * 1.0;
        let denominator = phi[0].powi(2) * 3.0 * 3.0 - phi[1].powi(2) * 1.0 * 1.0;

        numerator / denominator
    }

    use glam::DVec2;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    fn run_intersection_tests() {
        let file = File::open("/tmp/intersections").expect("Failed to open file");
        let reader = BufReader::new(file);

        let mut lines = reader.lines();
        let mut test_number = 0;
        let mut passes = 0;
        let mut fails = 0;
        let mut colinear = 0;

        while let Some(Ok(line1)) = lines.next() {
            test_number += 1;
            if line1.trim().is_empty() {
                continue;
            }

            let line2 = match lines.next() {
                Some(Ok(line)) => line,
                _ => continue,
            };

            let line3 = match lines.next() {
                Some(Ok(line)) => line,
                _ => continue,
            };

            // Parse curves
            let curve1 = parse_curve(&line1);
            let curve2 = parse_curve(&line2);

            if curve1.is_none() || curve2.is_none() {
                println!("Failed to parse curves for test {}", test_number);
                continue;
            }

            let curve1 = curve1.unwrap();
            let curve2 = curve2.unwrap();

            let (min, max) = compute_bounding_box(&[&curve1, &curve2]);
            let curve1 = normalize_curve(&curve1, min, max);
            let curve2 = normalize_curve(&curve2, min, max);

            let implicit1 = curve1.to_implicit();
            let implicit2 = curve2.to_implicit();
            // if implicit1.is_none() || implicit2.is_none() {
            // colinear += 1;
            // continue;
            // }

            // Parse expected intersections
            let expected_intersections: Vec<DVec2> = line3
                .split(' ')
                .map(|point| {
                    point
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect()
                })
                .map(|vec: Vec<_>| DVec2::new(vec[0], vec[1]))
                .collect();

            // Find intersections
            let Some(found_intersections) = find_intersections(&curve1, &curve2) else {
                colinear += 1;
                continue;
            };
            let found_intersections: Vec<_> = found_intersections.iter().map(|(_, x)| x).collect();

            // if expected_intersections.len() > 1 {
            //     println!("number_intersections: {}", expected_intersections.len());
            // }
            // continue;
            // Compare results
            println!("Test {}:", test_number);
            println!("  Expected intersections: {:?}", expected_intersections);
            println!("  Found intersections: {:?}", found_intersections);

            if expected_intersections.len() != found_intersections.len() {
                println!("  FAIL: Number of intersections doesn't match");
                println!(
                    " curve1: {:#?}\n curve2{:#?}",
                    curve1.control_points, curve2.control_points
                );
                fails += 1;
                continue;
            }

            let mut all_matched = true;
            for expected_t in expected_intersections {
                let expected_point = curve1.sample_at(expected_t.x);
                if !found_intersections
                    .iter()
                    .any(|&found_point| (found_point - expected_t).length() < 1e-6)
                {
                    all_matched = false;
                    println!(
                        "  FAIL: Expected intersection at t={} (point {:?}) not found",
                        expected_t, expected_point
                    );
                }
            }

            if all_matched {
                println!("  PASS");
                passes += 1;
            }
        }
        println!("total fails: {}", fails);
        println!("total passes: {}", passes);
        println!("total skipped because collinear: {}", colinear);
    }

    fn parse_curve(line: &str) -> Option<CubicBezier> {
        let coords: Vec<f64> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        if coords.len() != 8 {
            dbg!(line);
            return None;
        }

        Some(CubicBezier::new([
            DVec2::new(coords[0], coords[1]),
            DVec2::new(coords[2], coords[3]),
            DVec2::new(coords[4], coords[5]),
            DVec2::new(coords[6], coords[7]),
        ]))
    }

    // Add this to your main function or tests module
    #[test]
    fn test_intersections() {
        run_intersection_tests();
    }
}
