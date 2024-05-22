<class '__main__.DynUNetSkipLayer'> DynUNetSkipLayer(
  (downsample): UnetResBlock(
    (conv1): Convolution(
      (conv): Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
    )
    (conv2): Convolution(
      (conv): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
    )
    (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
    (norm1): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (norm2): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (conv3): Convolution(
      (conv): Conv3d(1, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
    )
    (norm3): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  )
  (next_layer): DynUNetSkipLayer(
    (downsample): UnetResBlock(
      (conv1): Convolution(
        (conv): Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
      )
      (conv2): Convolution(
        (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
      (norm1): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (conv3): Convolution(
        (conv): Conv3d(32, 64, kernel_size=(1, 1, 1), stride=(2, 2, 2), bias=False)
      )
      (norm3): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
    (next_layer): DynUNetSkipLayer(
      (downsample): UnetResBlock(
        (conv1): Convolution(
          (conv): Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
        )
        (conv2): Convolution(
          (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
        (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        (norm1): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (norm2): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (conv3): Convolution(
          (conv): Conv3d(64, 128, kernel_size=(1, 1, 1), stride=(2, 2, 2), bias=False)
        )
        (norm3): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
      (next_layer): DynUNetSkipLayer(
        (downsample): UnetResBlock(
          (conv1): Convolution(
            (conv): Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
          )
          (conv2): Convolution(
            (conv): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          (norm1): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (norm2): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (conv3): Convolution(
            (conv): Conv3d(128, 256, kernel_size=(1, 1, 1), stride=(2, 2, 2), bias=False)
          )
          (norm3): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        )
        (next_layer): DynUNetSkipLayer(
          (downsample): UnetResBlock(
            (conv1): Convolution(
              (conv): Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
            )
            (conv2): Convolution(
              (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            )
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
            (norm1): InstanceNorm3d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (norm2): InstanceNorm3d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (conv3): Convolution(
              (conv): Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(2, 2, 2), bias=False)
            )
            (norm3): InstanceNorm3d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
          (next_layer): UnetResBlock(
            (conv1): Convolution(
              (conv): Conv3d(512, 1024, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
            )
            (conv2): Convolution(
              (conv): Conv3d(1024, 1024, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            )
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
            (norm1): InstanceNorm3d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (norm2): InstanceNorm3d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (conv3): Convolution(
              (conv): Conv3d(512, 1024, kernel_size=(1, 1, 1), stride=(2, 2, 2), bias=False)
            )
            (norm3): InstanceNorm3d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
          (upsample): UnetUpBlock(
            (transp_conv): Convolution(
              (conv): ConvTranspose3d(1024, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
            )
            (conv_block): UnetBasicBlock(
              (conv1): Convolution(
                (conv): Conv3d(1024, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
              )
              (conv2): Convolution(
                (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
              )
              (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
              (norm1): InstanceNorm3d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
              (norm2): InstanceNorm3d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            )
          )

          #  --------------------------------------------------------

          # DEEP 1
          (enc_head): DeepUp(
            (conv_block): UnetOutBlock(
              (conv): Convolution(
                (conv): Conv3d(512, 13, kernel_size=(1, 1, 1), stride=(1, 1, 1))
              )
            )
            (transp_conv1): UpSample(
              (deconv): ConvTranspose3d(13, 13, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            )
          )

          # DEEP 0
          (dec_head): DeepUp(
            (conv_block): UnetOutBlock(
              (conv): Convolution(
                (conv): Conv3d(1024, 13, kernel_size=(1, 1, 1), stride=(1, 1, 1))
              )
            )
            (transp_conv1): UpSample(
              (deconv): ConvTranspose3d(13, 13, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            )
          )
        )

        # ---------------------------------------------------------------

        (upsample): UnetUpBlock(
          (transp_conv): Convolution(
            (conv): ConvTranspose3d(512, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
          )
          (conv_block): UnetBasicBlock(
            (conv1): Convolution(
              (conv): Conv3d(512, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            )
            (conv2): Convolution(
              (conv): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            )
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
            (norm1): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (norm2): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (enc_head): DeepUp(
          (conv_block): UnetOutBlock(
            (conv): Convolution(
              (conv): Conv3d(256, 13, kernel_size=(1, 1, 1), stride=(1, 1, 1))
            )
          )
          (transp_conv1): UpSample(
            (deconv): ConvTranspose3d(13, 13, kernel_size=(2, 2, 2), stride=(2, 2, 2))
          )
        )
        (dec_head): DeepUp(
          (conv_block): UnetOutBlock(
            (conv): Convolution(
              (conv): Conv3d(512, 13, kernel_size=(1, 1, 1), stride=(1, 1, 1))
            )
          )
          (transp_conv1): UpSample(
            (deconv): ConvTranspose3d(13, 13, kernel_size=(2, 2, 2), stride=(2, 2, 2))
          )
        )
      )
      # --------------------------------------------------------------------------------------------------------
      (upsample): UnetUpBlock(
        (transp_conv): Convolution(
          (conv): ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        )
        (conv_block): UnetBasicBlock(
          (conv1): Convolution(
            (conv): Conv3d(256, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
          (conv2): Convolution(
            (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          (norm1): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (norm2): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        )
      )
      (enc_head): DeepUp(
        (conv_block): UnetOutBlock(
          (conv): Convolution(
            (conv): Conv3d(128, 13, kernel_size=(1, 1, 1), stride=(1, 1, 1))
          )
        )
        (transp_conv1): UpSample(
          (deconv): ConvTranspose3d(13, 13, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
      )
      (dec_head): DeepUp(
        (conv_block): UnetOutBlock(
          (conv): Convolution(
            (conv): Conv3d(256, 13, kernel_size=(1, 1, 1), stride=(1, 1, 1))
          )
        )
        (transp_conv1): UpSample(
          (deconv): ConvTranspose3d(13, 13, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
      )
    )
    # --------------------------------------------------------------------------------------------------------
    (upsample): UnetUpBlock(
      (transp_conv): Convolution(
        (conv): ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
      )
      (conv_block): UnetBasicBlock(
        (conv1): Convolution(
          (conv): Conv3d(128, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
        (conv2): Convolution(
          (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
        (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        (norm1): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (norm2): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (enc_head): DeepUp(
      (conv_block): UnetOutBlock(
        (conv): Convolution(
          (conv): Conv3d(64, 13, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        )
      )
      (transp_conv1): UpSample(
        (deconv): ConvTranspose3d(13, 13, kernel_size=(2, 2, 2), stride=(2, 2, 2))
      )
    )
    (dec_head): DeepUp(
      (conv_block): UnetOutBlock(
        (conv): Convolution(
          (conv): Conv3d(128, 13, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        )
      )
      (transp_conv1): UpSample(
        (deconv): ConvTranspose3d(13, 13, kernel_size=(2, 2, 2), stride=(2, 2, 2))
      )
    )
  )
  # --------------------------------------------------------------------------------------------------------------------------------------------
  (upsample): UnetUpBlock(
    (transp_conv): Convolution(
      (conv): ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
    )
    (conv_block): UnetBasicBlock(
      (conv1): Convolution(
        (conv): Conv3d(64, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (conv2): Convolution(
        (conv): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      )
      (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
      (norm1): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (enc_head): DeepUp(
    (conv_block): UnetOutBlock(
      (conv): Convolution(
        (conv): Conv3d(32, 13, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      )
    )
    (transp_conv1): UpSample(
      (deconv): ConvTranspose3d(13, 13, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    )
  )
  (dec_head): DeepUp(
    (conv_block): UnetOutBlock(
      (conv): Convolution(
        (conv): Conv3d(64, 13, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      )
    )
    (transp_conv1): UpSample(
      (deconv): ConvTranspose3d(13, 13, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    )
  )
)
