## Sample code to convert a Pytorch DNN model with the Sony DNN Compiler 

This sample code is *not* intended to be a real functional DNN, but it shows an example of how a 
state tensor that persists from frame-to-frame can be introduced to a DNN.

### Python Code

In `SampleModel.__init__()` in model.py, a state tensor is created and initialized using one of the 
following methods:

```python
  self.state = torch.zeros(32, 32)  # initialized to all zeros
  self.state = torch.rand(32, 32)  # initialized to random data
```

In `SampleModel.forward()`, the state tensor is used as an input to an add layer. Then, the output 
of the add layer is used to update the state tensor for the next frame:

```python
    # state = x + state
    if self.quantize:
        x = self.dequant_2(x)
        x = torch.add(x, self.state)
        x = self.quant_2(x)        # Classification layer
    else:
        x = torch.add(x, self.state)
    self.state = x
```

### DNN Compiler Configuration File

A sample configuration file for this model has been provided in 
`configs\imx681_pytorch_classification_with_state_i2c.cfg`

By default, the DNN compiler will incorrectly create two separate tensors for the state: one for 
when it is read and one for when it is written. In order to tell the compiler to combine these and 
treat them as a single tensor, the following parameters must be set in the configuration file:

MODEL_STATE_READ_BUFFER_ID - the buffer ID (from summary.txt) of the tensor that uses the state as an input
MODEL_STATE_WRITE_BUFFER_ID - the buffer ID (from summary.txt) of the tensor that updates the state as an output

When developing a DNN with a state tensor, the following flow should be followed:

First, compile the model *without* setting `MODEL_STATE_READ_BUFFER_ID` and `MODEL_STATE_WRITE_BUFFER_ID`.
As an example, you can comment out the following lines in `imx681_pytorch_classification_with_state_i2c.cfg`:

```
# MODEL_STATE_READ_BUFFER_ID 27
# MODEL_STATE_WRITE_BUFFER_ID 28
```

Then, look at the summary.txt output file that is generated. Find the layers that are using your 
state tensor and note the buffer IDs for where it is read and written. As an example, the following 
section of the summary.txt file describes the add operation (state = x + state) in this DNN:

```
OPERATION 16: ADDSUB (Working Mem: 0x002bafe0, Params: 0x00000764)
  Parameters:
    subtract=False, mode=MATRIX, scale_ratio_a=0.0073, scale_ratio_b=25.2095, offset=-127.4923
  Inputs:
    ID      Dimensions            Type    Location      Address   Data Start  Data Size   Row Stride  Bat Stride  Parent  # Con   Quant Type    QS[0]   QZ[0]   
    26      1 x 1 x 32 x 1        S CHAR  SCRATCH_RAM   0x2bb4e0  (0, 0)      1 x 32      32          32          -1      2       PER_TENSOR    0.00029 -70     
    27      1 x 32 x 32 x 1       S CHAR  STATIC_DATA   0x297480  (0, 0)      32 x 32     32          1024        -1      1       PER_TENSOR    1.00000 0       
  Outputs:
    ID      Dimensions            Type    Location      Address   Data Start  Data Size   Row Stride  Bat Stride  Parent  # Con   Quant Type    QS[0]   QZ[0]   
    28      1 x 32 x 32 x 1       S CHAR  SCRATCH_RAM   0x2bb0e0  (0, 0)      32 x 32     32          1024        -1      2       PER_TENSOR    0.03967 -128    
  Est. Processing Time: 0.001 ms
  ```

Buffer ID 27 is the 2nd input to the add operation and represents the *read* of the state tensor.
Buffer ID 28 is the output of the add operation and represents the *write* of the state tensor.

After finding this information, add `MODEL_STATE_READ_BUFFER_ID` and `MODEL_STATE_WRITE_BUFFER_ID` 
to your configuration file and re-compile. In this case, this is accomplished by uncommenting these 
two lines in the sample configuration file:

```
MODEL_STATE_READ_BUFFER_ID 27
MODEL_STATE_WRITE_BUFFER_ID 28
```

Now, the DNN compiler will treat these two buffers as part of the same tensor and the state tensor
will be implemented correctly.

Here is an example of what the updated summary.txt looks like in this case:

```
OPERATION 16: ADDSUB (Working Mem: 0x002bafe0, Params: 0x00000764)
  Parameters:
    subtract=False, mode=MATRIX, scale_ratio_a=0.0073, scale_ratio_b=25.2095, offset=-127.4923
  Inputs:
    ID      Dimensions            Type    Location      Address   Data Start  Data Size   Row Stride  Bat Stride  Parent  # Con   Quant Type    QS[0]   QZ[0]   
    26      1 x 1 x 32 x 1        S CHAR  SCRATCH_RAM   0x2bb4e0  (0, 0)      1 x 32      32          32          -1      2       PER_TENSOR    0.00029 -70     
    27      1 x 32 x 32 x 1       S CHAR  STATIC_DATA   0x297480  (0, 0)      32 x 32     32          1024        -1      1       PER_TENSOR    0.03967 -128    
  Outputs:
    ID      Dimensions            Type    Location      Address   Data Start  Data Size   Row Stride  Bat Stride  Parent  # Con   Quant Type    QS[0]   QZ[0]   
    28      1 x 32 x 32 x 1       S CHAR  STATIC_DATA   0x297480  (0, 0)      32 x 32     32          1024        -1      2       PER_TENSOR    0.03967 -128    
  Est. Processing Time: 0.001 ms
```

Note that 3 major changes have been made:
1. Buffer 27 and 28 now point to the same memory address (both use 0x297480)
2. Buffer 27 and 28 are now both located in the STATIC_DATA region
3. Buffer 27 and 28 now use the same quantization parameters (QS=0.03967 and QZ=-128)

These 3 updates are a good check to confirm that the compiler is handling the state tensor as intended.


### To train the model from scratch:
```
python train.py --batch_size 32 --num_epochs 10 --learning_rate 0.001 --use_cuda
```

### To run the Sony DNN Compiler on a trained model:
```
python load_and_compile.py
```

