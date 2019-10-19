package com.arb222.lesionclassifier;

import com.google.gson.annotations.SerializedName;

public class Output {
    @SerializedName("category")
    private int category;

    @SerializedName("probability")
    private double prediction;

    public int getCategory() {
        return category;
    }

    public double getPrediction() {
        return prediction;
    }

    public Output(int category, double prediction) {
        this.category = category;
        this.prediction = prediction;
    }
}
