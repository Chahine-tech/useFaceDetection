use opencv::{highgui, imgcodecs, imgproc, objdetect, prelude::*, types};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let face_cascade_path = "data/haarcascade_frontalface_default.xml";
    // Declare `face_cascade` as mutable
    let mut face_cascade = objdetect::CascadeClassifier::new(face_cascade_path)?;

    let image_path = "data/image.jpg";
    let mut img = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;

    let mut faces = types::VectorOfRect::new();
    // Now `face_cascade` can be borrowed as mutable
    face_cascade.detect_multi_scale(
        &img,
        &mut faces,
        1.1,
        3,
        objdetect::CASCADE_SCALE_IMAGE,
        opencv::core::Size::new(30, 30),
        opencv::core::Size::default(),
    )?;

    for face in faces {
        imgproc::rectangle(
            &mut img,
            face,
            opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
            imgproc::LINE_8,
            0,
        )?;
    }

    highgui::named_window("Face Detection", highgui::WINDOW_AUTOSIZE)?;
    highgui::imshow("Face Detection", &img)?;
    highgui::wait_key(0)?;

    Ok(())
}
