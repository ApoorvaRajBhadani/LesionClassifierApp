package com.arb222.lesionclassifier;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.theartofdev.edmodo.cropper.CropImage;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class ImageSelectionActivity extends AppCompatActivity {

    ImageView selectedImageImageView;
    Button selectImageButton;
    TextView outputTextView;

    Uri mImageUri;
    Bitmap mImageBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_selection);

        selectedImageImageView = (ImageView) findViewById(R.id.imageView);
        selectImageButton = (Button) findViewById(R.id.select_image_button);
        outputTextView = (TextView) findViewById(R.id.output_textview);

        selectImageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onChooseFile();
            }
        });
    }

    public void onChooseFile() {
        CropImage.activity().start(ImageSelectionActivity.this);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE) {
            CropImage.ActivityResult result = CropImage.getActivityResult(data);

            if (resultCode == RESULT_OK) {
                mImageUri = result.getUri();
                selectedImageImageView.setImageURI(mImageUri);
               // mImageBitmap = result.getBitmap();
                mImageBitmap = ((BitmapDrawable) selectedImageImageView.getDrawable()).getBitmap();

                File filesDir = getApplicationContext().getCacheDir();
                File imageFile = new File(filesDir, "image" + ".jpg");

                OutputStream os;
                try {
                    os = new FileOutputStream(imageFile);
                    mImageBitmap.compress(Bitmap.CompressFormat.JPEG, 100, os);
                    os.flush();
                    os.close();
                } catch (Exception e) {
                    Log.e(getClass().getSimpleName(), "Error writing bitmap", e);
                }

                RequestBody requestBitmap = RequestBody.create(MediaType.parse("multipart/form-data"),imageFile);
                MultipartBody.Part body = MultipartBody.Part.createFormData("image",imageFile.getName(),requestBitmap);



                Retrofit retrofit = new Retrofit.Builder()
                        .baseUrl("http://192.168.43.19:5000/")
                        .addConverterFactory(GsonConverterFactory.create())
                        .build();

                PredictionApi predictionApi = retrofit.create(PredictionApi.class);

                Call<Output> call = predictionApi.createPost(body);

                call.enqueue(new Callback<Output>() {
                    @Override
                    public void onResponse(Call<Output> call, Response<Output> response) {
                        if (!response.isSuccessful()) {
                            outputTextView.setText("code: " + response.code());
                            return;
                        }

                        Output postResponse = response.body();

                        String content = "";
                        content += "code: " + response.code() + "\n";
                        content += "category: " + postResponse.getCategory() + "\n";
                        content += "prediction: " + postResponse.getPrediction();

                        outputTextView.setText(content);
                    }

                    @Override
                    public void onFailure(Call<Output> call, Throwable t) {
                        outputTextView.setText(t.getMessage());
                        Toast.makeText(ImageSelectionActivity.this,"Error",Toast.LENGTH_SHORT).show();
                    }
                });

            } else if (resultCode == CropImage.CROP_IMAGE_ACTIVITY_RESULT_ERROR_CODE) {
                Exception e = result.getError();
                Toast.makeText(this, "Error : " + e, Toast.LENGTH_SHORT).show();
            }


        }
    }
}
