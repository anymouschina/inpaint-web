/* eslint-disable no-console */
/* eslint-disable no-plusplus */
import cv, { Mat } from 'opencv-ts'
import { getCapabilities } from './util'
import { ensureModel } from './cache'

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'Anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`Failed to load image from ${url}`))
    img.src = url
  })
}
function imgProcess(img: Mat) {
  const channels = new cv.MatVector()
  cv.split(img, channels) // 分割通道

  const C = channels.size() // 通道数
  const H = img.rows // 图像高度
  const W = img.cols // 图像宽度

  const chwArray = new Float32Array(C * H * W) // 创建新的数组来存储转换后的数据

  for (let c = 0; c < C; c++) {
    const channelData = channels.get(c).data // 获取单个通道的数据
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        chwArray[c * H * W + h * W + w] = channelData[h * W + w] / 255.0
        // chwArray[c * H * W + h * W + w] = channelData[h * W + w]
      }
    }
  }

  channels.delete() // 清理内存
  return chwArray // 返回转换后的数据
}
/**
 * 图像超分辨率分块处理函数
 * 将大图像分割成较小的块进行处理，以便在内存有限情况下处理大图像
 * 每个块由模型处理后放大4倍，然后重新组合成完整输出图像
 * 
 * @param inputTensor - 输入图像张量，格式为 CHW (通道-高度-宽度)
 * @param session - ONNX 推理会话，用于执行超分辨率模型
 * @param callback - 进度回调函数，用于报告处理进度
 * @returns 处理后的超分辨率图像张量
 */
async function tileProc(
  inputTensor: ort.Tensor,
  session: ort.InferenceSession,
  callback: (progress: number) => void
) {
  // 获取输入图像的尺寸信息
  const inputDims = inputTensor.dims
  const imageW = inputDims[3]  // 图像宽度
  const imageH = inputDims[2]  // 图像高度

  // 计算RGB通道在一维数组中的偏移量
  const rOffset = 0                  // R通道的偏移量
  const gOffset = imageW * imageH    // G通道的偏移量
  const bOffset = imageW * imageH * 2 // B通道的偏移量

  // 计算输出图像的尺寸 (所有维度不变，但高度和宽度放大4倍)
  const outputDims = [
    inputDims[0],
    inputDims[1],
    inputDims[2] * 4,  // 高度放大4倍
    inputDims[3] * 4,  // 宽度放大4倍
  ]
  
  // 创建输出张量
  const outputTensor = new ort.Tensor(
    'float32',
    new Float32Array(
      outputDims[0] * outputDims[1] * outputDims[2] * outputDims[3]
    ),
    outputDims
  )

  // 计算输出图像的尺寸和通道偏移量
  const outImageW = outputDims[3]
  const outImageH = outputDims[2]
  const outROffset = 0                   // 输出R通道偏移量
  const outGOffset = outImageW * outImageH    // 输出G通道偏移量
  const outBOffset = outImageW * outImageH * 2 // 输出B通道偏移量

  // 定义分块处理参数
  const tileSize = 64         // 每个处理块的大小 (64x64像素)
  const tilePadding = 6       // 每个块四周的填充大小，用于减少块之间的边界效应
  const tileSizePre = tileSize - tilePadding * 2  // 块的有效内容大小 (52x52像素)

  // 计算水平和垂直方向需要的块数
  const tilesx = Math.ceil(inputDims[3] / tileSizePre)  // 水平块数
  const tilesy = Math.ceil(inputDims[2] / tileSizePre)  // 垂直块数

  // 获取输入数据
  const { data } = inputTensor

  console.log(inputTensor)
  const numTiles = tilesx * tilesy  // 总块数
  let currentTile = 0  // 当前处理的块索引

  // 逐块处理图像
  for (let i = 0; i < tilesx; i++) {
    for (let j = 0; j < tilesy; j++) {
      const ti = Date.now()  // 记录当前块处理开始时间
      
      // 计算当前块的实际尺寸（边缘可能不足一个完整块）
      const tileW = Math.min(tileSizePre, imageW - i * tileSizePre)
      const tileH = Math.min(tileSizePre, imageH - j * tileSizePre)
      console.log(`tileW: ${tileW} tileH: ${tileH}`)
      
      // 定义当前块在RGB三通道的偏移量
      const tileROffset = 0
      const tileGOffset = tileSize * tileSize
      const tileBOffset = tileSize * tileSize * 2

      // 创建当前块的数据数组，包含RGB三通道
      const tileData = new Float32Array(tileSize * tileSize * 3)
      
      // 从输入图像提取当前块，包含填充区域
      for (let xp = -tilePadding; xp < tileSizePre + tilePadding; xp++) {
        for (let yp = -tilePadding; yp < tileSizePre + tilePadding; yp++) {
          // 计算在原始图像中的实际坐标，并处理边界情况
          let xim = i * tileSizePre + xp
          if (xim < 0) xim = 0                // 边界处理：左边界
          else if (xim >= imageW) xim = imageW - 1  // 边界处理：右边界

          // 计算在原始图像中的实际坐标，并处理边界情况
          let yim = j * tileSizePre + yp
          if (yim < 0) yim = 0                // 边界处理：上边界
          else if (yim >= imageH) yim = imageH - 1  // 边界处理：下边界

          const idx = xim + yim * imageW  // 在一维数组中的索引

          // 计算在块数据中的坐标
          const xt = xp + tilePadding
          const yt = yp + tilePadding
          
          // 复制RGB三通道像素值到块数据中
          tileData[xt + yt * tileSize + tileROffset] = data[idx + rOffset]  // R通道
          tileData[xt + yt * tileSize + tileGOffset] = data[idx + gOffset]  // G通道
          tileData[xt + yt * tileSize + tileBOffset] = data[idx + bOffset]  // B通道
        }
      }

      // 创建用于模型输入的张量
      const tile = new ort.Tensor('float32', tileData, [
        1,  // 批次大小
        3,  // 通道数 (RGB)
        tileSize,  // 高度
        tileSize,  // 宽度
      ])
      
      // 运行模型推理，增强当前块
      const r = await session.run({ 'input.1': tile })
      const results = {
        output: r['1895'],  // 获取模型输出结果
      }
      console.log(`pre dims:${results.output.dims}`)

      // 计算输出块的尺寸参数 (尺寸放大4倍)
      const outTileW = tileW * 4  // 输出块宽度
      const outTileH = tileH * 4  // 输出块高度
      const outTileSize = tileSize * 4  // 输出块大小
      const outTileSizePre = tileSizePre * 4  // 输出块有效内容大小

      // 定义输出块的RGB通道偏移量
      const outTileROffset = 0
      const outTileGOffset = outTileSize * outTileSize
      const outTileBOffset = outTileSize * outTileSize * 2

      // 将处理后的块合并到输出张量中，去除填充区域
      for (let x = 0; x < outTileW; x++) {
        for (let y = 0; y < outTileH; y++) {
          // 计算在输出图像中的位置
          const xim = i * outTileSizePre + x
          const yim = j * outTileSizePre + y
          const idx = xim + yim * outImageW  // 在输出一维数组中的索引
          
          // 计算在当前输出块中的位置 (加上填充偏移)
          const xt = x + tilePadding * 4
          const yt = y + tilePadding * 4
          
          // 复制RGB三通道的值到输出张量
          outputTensor.data[idx + outROffset] =
            results.output.data[xt + yt * outTileSize + outTileROffset]  // R通道
          outputTensor.data[idx + outGOffset] =
            results.output.data[xt + yt * outTileSize + outTileGOffset]  // G通道
          outputTensor.data[idx + outBOffset] =
            results.output.data[xt + yt * outTileSize + outTileBOffset]  // B通道
        }
      }
      
      // 更新进度
      currentTile++
      const dt = Date.now() - ti  // 计算当前块处理耗时
      const remTime = (numTiles - currentTile) * dt  // 估计剩余时间
      console.log(
        `tile ${currentTile} of ${numTiles} took ${dt} ms, remaining time: ${remTime} ms`
      )
      callback(Math.round(100 * (currentTile / numTiles)))  // 调用进度回调函数
    }
  }
  console.log(`output dims:${outputTensor.dims}`)
  return outputTensor  // 返回处理完成的输出张量
}
function processImage(
  img: HTMLImageElement,
  canvasId?: string
): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    try {
      const src = cv.imread(img)
      // eslint-disable-next-line camelcase
      const src_rgb = new cv.Mat()
      // 将图像从RGBA转换为RGB
      cv.cvtColor(src, src_rgb, cv.COLOR_RGBA2RGB)
      if (canvasId) {
        cv.imshow(canvasId, src_rgb)
      }
      resolve(imgProcess(src_rgb))

      src.delete()
      src_rgb.delete()
    } catch (error) {
      reject(error)
    }
  })
}
function configEnv(capabilities: {
  webgpu: any
  wasm?: boolean
  simd: any
  threads: any
}) {
  ort.env.wasm.wasmPaths =
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/'
  if (capabilities.webgpu) {
    ort.env.wasm.numThreads = 1
  } else {
    if (capabilities.threads) {
      ort.env.wasm.numThreads = navigator.hardwareConcurrency ?? 4
    }
    if (capabilities.simd) {
      ort.env.wasm.simd = true
    }
    ort.env.wasm.proxy = true
  }
  console.log('env', ort.env.wasm)
}
function postProcess(floatData: Float32Array, width: number, height: number) {
  const chwToHwcData = []
  const size = width * height

  for (let h = 0; h < height; h++) {
    for (let w = 0; w < width; w++) {
      for (let c = 0; c < 3; c++) {
        // RGB通道
        const chwIndex = c * size + h * width + w
        const pixelVal = floatData[chwIndex]
        let newPiex = pixelVal
        if (pixelVal > 1) {
          newPiex = 1
        } else if (pixelVal < 0) {
          newPiex = 0
        }
        chwToHwcData.push(newPiex * 255) // 归一化反转
      }
      chwToHwcData.push(255) // Alpha通道
    }
  }
  return chwToHwcData
}

function imageDataToDataURL(imageData: ImageData) {
  // 创建 canvas
  const canvas = document.createElement('canvas')
  canvas.width = imageData.width
  canvas.height = imageData.height

  // 绘制 imageData 到 canvas
  const ctx = canvas.getContext('2d')
  ctx.putImageData(imageData, 0, 0)

  // 导出为数据 URL
  return canvas.toDataURL()
}
let model: ArrayBuffer | null = null
export default async function superResolution(
  imageFile: File | HTMLImageElement,
  callback: (progress: number) => void
) {
  console.time('sessionCreate')
  if (!model) {
    const capabilities = await getCapabilities()
    configEnv(capabilities)
    const modelBuffer = await ensureModel('superResolution')
    model = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: [capabilities.webgpu ? 'webgpu' : 'wasm'],
    })
  }
  console.timeEnd('sessionCreate')

  const img =
    imageFile instanceof HTMLImageElement
      ? imageFile
      : await loadImage(URL.createObjectURL(imageFile))
  const imageTersorData = await processImage(img)
  const imageTensor = new ort.Tensor('float32', imageTersorData, [
    1,
    3,
    img.height,
    img.width,
  ])

  const result = await tileProc(imageTensor, model, callback)
  console.time('postProcess')
  const outsTensor = result
  const chwToHwcData = postProcess(
    outsTensor.data,
    img.width * 4,
    img.height * 4
  )
  const imageData = new ImageData(
    new Uint8ClampedArray(chwToHwcData),
    img.width * 4,
    img.height * 4
  )
  console.log(imageData, 'imageData')
  const url = imageDataToDataURL(imageData)
  console.timeEnd('postProcess')

  return url
}
