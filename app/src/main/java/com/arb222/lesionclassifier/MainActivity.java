package com.arb222.lesionclassifier;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {

    Button scanSkinButton,findDoctorsButton,readBlogsButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        scanSkinButton = (Button)findViewById(R.id.add_image_button);
        findDoctorsButton = (Button)findViewById(R.id.nearby_doctors_button);
        readBlogsButton = (Button) findViewById(R.id.medi_blogs_button);

        scanSkinButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(MainActivity.this,ImageSelectionActivity.class);
                startActivity(i);
            }
        });
    }
}
