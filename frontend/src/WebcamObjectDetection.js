import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as cocossd from "@tensorflow-models/coco-ssd";
import './WebcamObjectDetection.css';

function WebcamObjectDetection() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  let imageLinkRef = useRef(null)

  useEffect(() => {
    async function startObjectDetection() {
    try {
        const video = videoRef.current;
        const canvas = canvasRef.current;

        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        await video.play();

        // Wait for the video to load metadata (including dimensions)
        await video.onloadedmetadata;

        // Set video dimensions
        video.width = video.videoWidth;
        video.height = video.videoHeight;

        // Set canvas dimensions
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const model = await cocossd.load();

        video.srcObject = stream;
        await video.play();

        function drawRect(ctx, x, y, width, height, label) {
            ctx.beginPath();
            ctx.rect(x, y, width, height);
            ctx.lineWidth = 2;
            ctx.strokeStyle = "red";
            ctx.fillStyle = "red";
            ctx.stroke();
          
            // Set a larger font size
            ctx.font = "24px Arial"; // You can adjust the font size and font family here
            ctx.fillText(label, x, y > 20 ? y - 10 : 20); // Adjust the y-coordinate for text placement
          }          

        async function detect() {
          const predictions = await model.detect(video);
          const context = canvas.getContext("2d");
          context.clearRect(0, 0, canvas.width, canvas.height);

          predictions.forEach((prediction) => {
            // if (prediction.class === 'apple' || prediction.class === 'banana') {
            //   const widthFactor = prediction.class === 'apple' ? 1.75 : 2.25
            //   if (!imageLinkRef.current) {
            //     canvasRef.current
            //       .getContext("2d")
            //       .drawImage(
            //         video,
            //         prediction.bbox[0], 
            //         prediction.bbox[1],
            //         prediction.bbox[2],
            //         prediction.bbox[3],
            //         0,
            //         0,
            //         prediction.bbox[2] * widthFactor,
            //         prediction.bbox[3]
            //       );
            //       canvasRef.current.toBlob((blob) => {
            //         imageLinkRef.current = blob
            //         makeAPIRequest(prediction.class)
            //         console.log('api request sent')
            //     })
            //     // setIsScanActive(false)
            //   }
            // }


            if (prediction.class === 'apple' || prediction.class === 'banana') {
              // Assuming `imageDataURL` contains the image data in Base64 format
              const imageDataURL = captureImage(prediction.bbox);
              
              // Create a Blob from the Base64 data
              const imageBlob = dataURLToBlob(imageDataURL);
              imageLinkRef.current = imageBlob
              
              // Create a FormData object and append the image
              const formData = new FormData();
              formData.append('image', imageBlob);
              
              makeAPIRequest(prediction.class);
            }

            // Capture the image data from the video using the bounding box
            function captureImage(bbox) {
              const canvas = document.createElement('canvas');
              const ctx = canvas.getContext('2d');
              
              canvas.width = bbox[2];
              canvas.height = bbox[3];
              
              canvasRef.current.getContext("2d").drawImage(
                video,
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                0,
                0,
                bbox[2],
                bbox[3]
              );
              
              return canvas.toDataURL('image/jpeg');
            }

            // Convert a data URL to a Blob
            function dataURLToBlob(dataURL) {
              const byteString = atob(dataURL.split(',')[1]);
              const mimeString = dataURL.split(',')[0].split(':')[1].split(';')[0];
              
              const ab = new ArrayBuffer(byteString.length);
              const ia = new Uint8Array(ab);
              
              for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
              }
              
              return new Blob([ab], { type: mimeString });
            }

            drawRect(
              context,
              prediction.bbox[0],
              prediction.bbox[1],
              prediction.bbox[2],
              prediction.bbox[3],
              `${prediction.class} (${Math.round(prediction.score * 100)}%)`
            );
          });

          requestAnimationFrame(detect);
        }

        detect();
      } catch (error) {
        console.error("Error in object detection:", error);
      }
    }

    startObjectDetection();
  }, []);

  const makeAPIRequest = async (fruitName) => {
    const formData = new FormData()
    formData.append("image", imageLinkRef.current)

    const response = await fetch("http://127.0.0.1:5000/classify/".concat(fruitName), {
      method: "POST",
      body: formData,
    });

    if (response.status === 200) {
      const text = await response.text()
      console.log(text)
    } else {
      console.log("Error with POST request")
    }
  }

  return (
    <div className="video-container">
      <video ref={videoRef} autoPlay playsInline muted className="video" />
      <canvas ref={canvasRef} className="overlay-canvas" />
    </div>
  );  
}

export default WebcamObjectDetection;
