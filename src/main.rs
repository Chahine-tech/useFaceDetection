use opencv::{core, highgui, imgproc, objdetect, prelude::*, videoio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let face_cascade_path = "data/haarcascade_frontalface_default.xml";
    let mut face_cascade = objdetect::CascadeClassifier::new(face_cascade_path)?;

    // Ouvrir le flux vidéo
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?; // 0 correspond à la première caméra
    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("Impossible d'ouvrir la caméra");
    }

    loop {
        let mut frame = core::Mat::default();
        cam.read(&mut frame)?;

        if frame.size()?.width > 0 {
            let mut faces = core::Vector::<core::Rect>::new();
            face_cascade.detect_multi_scale(
                &frame,
                &mut faces,
                1.1,
                3,
                objdetect::CASCADE_SCALE_IMAGE,
                opencv::core::Size::new(30, 30),
                opencv::core::Size::default(),
            )?;

            for face in faces {
                imgproc::rectangle(
                    &mut frame,
                    face,
                    opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                    1,
                    imgproc::LINE_8,
                    0,
                )?;
            }

            highgui::imshow("Face Detection", &frame)?;
        }

        // Quitter la boucle si la touche 'q' est pressée
        if highgui::wait_key(10)? == 113 {
            break;
        }
    }

    Ok(())
}
