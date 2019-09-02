# README

## Requirements
Python 3.7

CUDA 10.0

PyTorch 1.1


## Trouble shooting

Pytorch does not have `same padding`, do this:

### Step 1: 
Go to this file:
`/venv/lib/python3.7/site-packages/torch/nn/modules/conv.py`

### Step 2:
Modify `forward` function in `class Conv2d( _ConvNd)`

    class Conv2d( _ConvNd):

        @weak_script_method
        def forward(self, input):
            #return F.conv2d(input, self.weight, self.bias, self.stride,
            #                        self.padding, self.dilation, self.groups)
            return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups) ## ZZ: same padding like TensorFlow

### Step 3: Add custom function
custom `con2d`, because pytorch don't have "padding='same'" option.
    
    def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):

        input_rows = input.size(2)
        filter_rows = weight.size(2)
        effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
        out_rows = (input_rows + stride[0] - 1) // stride[0]
        padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows -
                  input_rows)
        padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
        rows_odd = (padding_rows % 2 != 0)
        padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
        cols_odd = (padding_rows % 2 != 0)

        if rows_odd or cols_odd:
            input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)
    



## How to Train?
`main_model_parallel.py`

## Output
                ./log
                ./plot
                ./checkpoints
                ./weights

## Evaluation
https://github.com/rafaelpadilla/Object-Detection-Metrics


## Change Log
1. Activation function

As the author mentioned:

``We use a linear activation for the final layer and all other layers use the leaky
rectified linear activation.``

