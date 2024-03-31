package com.example.resnetapp


import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.resnetapp.ml.LiteModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer


class MainActivity : AppCompatActivity() {


    private lateinit var selectBtn: Button
    private lateinit var predBtn: Button
    private lateinit var resView: TextView
    private lateinit var imageView: ImageView
    private lateinit var bitmap: Bitmap



    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        selectBtn = findViewById(R.id.selectBtn)
        predBtn = findViewById(R.id.predictBtn)
        resView = findViewById(R.id.resView)
        imageView = findViewById(R.id.imageView)


        val labels = application.assets.open("labels.txt").bufferedReader().readLines()
        bitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)

        //image processor
//        val imageProcessor = ImageProcessor.Builder()
//            .add(ResizeOp(1, 1, ResizeOp.ResizeMethod.BILINEAR))
////            .add(NormalizeOp(0.0f, 255.0f))
//            .build()


// Resize the image to match the input size expected by the model (224x224 pixels)
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .build()
        val tensorImage = TensorImage.fromBitmap(bitmap)


// Convert the resized image to a float array


        selectBtn.setOnClickListener {
            val intent = Intent()
            intent.setAction(Intent.ACTION_GET_CONTENT)
            intent.setType("image/*")
            startActivityForResult(intent, 100)
        }



        predBtn.setOnClickListener {
            if (!::bitmap.isInitialized) {
                // Handle the case where bitmap is not initialized
                return@setOnClickListener
            }

            try {
                var tensorImage = TensorImage(DataType.FLOAT32)
                tensorImage.load(bitmap)

                tensorImage = imageProcessor.process(tensorImage)

                //tensorImage = reshapedImgArray.process(tensorImage)


                val model = LiteModel.newInstance(this)

// Creates inputs for reference.
                val inputFeature0 =
                    TensorBuffer.createFixedSize(intArrayOf(1, 224 , 224, 3), DataType.FLOAT32)
                inputFeature0.loadBuffer(tensorImage.buffer)

// Runs model inference and gets result.
                try {
                    val outputs = model.process(inputFeature0)
                    val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray
                    val maxIdx = outputFeature0.indices.maxByOrNull { outputFeature0[it] } ?: -1
//

                    resView.text = labels[maxIdx]


                } catch (e: Exception) {
                    Log.e("TFLite", "Model inference error:", e) // Log error details
                    // Display an error message to the user
                    resView.text = "Error during prediction. Check logs for details."

                } finally {
                    model.close()
                }
            } catch (e: Exception) {
                Log.e("TFLite", "Prediction error:", e)
                // Display a general error message
                resView.text = "An error occurred during prediction."

            }

        }


    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == 100 && resultCode == RESULT_OK) {
            var uri = data?.data;
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            imageView.setImageBitmap(bitmap)

        }
    }
}




