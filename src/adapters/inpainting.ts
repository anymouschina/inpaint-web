// @ts-nocheck
/* eslint-disable camelcase */
/* eslint-disable no-plusplus */
import cv, { Mat } from 'opencv-ts'
import { ensureModel } from './cache'
import { getCapabilities } from './util'
import type { modelType } from './cache'
// ort.env.debug = true
// ort.env.logLevel = 'verbose'
// ort.env.webgpu.profilingMode = 'default'

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

  const chwArray = new Uint8Array(C * H * W) // 创建新的数组来存储转换后的数据

  for (let c = 0; c < C; c++) {
    const channelData = channels.get(c).data // 获取单个通道的数据
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        chwArray[c * H * W + h * W + w] = channelData[h * W + w]
        // chwArray[c * H * W + h * W + w] = channelData[h * W + w]
      }
    }
  }

  channels.delete() // 清理内存
  return chwArray // 返回转换后的数据
}
function markProcess(img: Mat) {
  const channels = new cv.MatVector()
  cv.split(img, channels) // 分割通道

  const C = 1 // 通道数
  const H = img.rows // 图像高度
  const W = img.cols // 图像宽度

  const chwArray = new Uint8Array(C * H * W) // 创建新的数组来存储转换后的数据

  for (let c = 0; c < C; c++) {
    const channelData = channels.get(0).data // 获取单个通道的数据
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        chwArray[c * H * W + h * W + w] = (channelData[h * W + w] !== 255) * 255
      }
    }
  }

  channels.delete() // 清理内存
  return chwArray // 返回转换后的数据
}
function processImage(
  img: HTMLImageElement,
  canvasId?: string
): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    try {
      const src = cv.imread(img)
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

function processMark(
  img: HTMLImageElement,
  canvasId?: string
): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    try {
      const src = cv.imread(img)
      const src_grey = new cv.Mat()

      // 将图像从RGBA转换为二值化
      cv.cvtColor(src, src_grey, cv.COLOR_BGR2GRAY)

      if (canvasId) {
        cv.imshow(canvasId, src_grey)
      }

      resolve(markProcess(src_grey))

      src.delete()
    } catch (error) {
      reject(error)
    }
  })
}
function postProcess(uint8Data: Uint8Array, width: number, height: number) {
  const chwToHwcData = []
  const size = width * height

  for (let h = 0; h < height; h++) {
    for (let w = 0; w < width; w++) {
      for (let c = 0; c < 3; c++) {
        // RGB通道
        const chwIndex = c * size + h * width + w
        const pixelVal = uint8Data[chwIndex]
        let newPiex = pixelVal
        if (pixelVal > 255) {
          newPiex = 255
        } else if (pixelVal < 0) {
          newPiex = 0
        }
        chwToHwcData.push(newPiex) // 归一化反转
      }
      chwToHwcData.push(255) // Alpha通道
    }
  }
  return chwToHwcData
}

function imageDataToDataURL(imageData) {
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

function configEnv(capabilities) {
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
const resizeMark = (
  image: HTMLImageElement,
  width: number,
  height: number
): Promise<HTMLImageElement> => {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height

    // 将图片绘制到canvas上，并调整大小
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      reject(new Error('Unable to get canvas context'))
      return
    }
    ctx.drawImage(image, 0, 0, width, height)

    // 获取调整大小后的图片URL
    const resizedImageUrl = canvas.toDataURL()

    // 创建一个新的Image对象并设置其src为调整大小后的图片URL
    const resizedImage = new Image()
    resizedImage.onload = () => resolve(resizedImage)
    resizedImage.onerror = () =>
      reject(new Error('Failed to load resized image'))
    resizedImage.src = resizedImageUrl
  })
}
let model: ArrayBuffer | null = null
/**
 * 图像修复/填充主函数
 * 该函数使用AI模型对图像进行修复，根据提供的蒙版填充图像中的缺失或需要修改的区域
 * 
 * @param imageFile - 输入图像，可以是文件对象或HTMLImageElement
 * @param maskBase64 - 蒙版图像的Base64编码字符串，白色区域表示需要修复的部分
 * @returns 修复后的图像的Data URL
 */
export default async function inpaint(
  imageFile: File | HTMLImageElement,
  maskBase64: string
) {
  console.time('sessionCreate')
  if (!model) {
    // 检测设备能力（WebGPU、WASM等）
    const capabilities = await getCapabilities()
    // 根据设备能力配置ONNX运行环境
    configEnv(capabilities)
    // 加载修复模型
    const modelBuffer = await ensureModel('inpaint')
    // 创建推理会话，优先使用WebGPU加速，不支持则使用WASM
    model = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: [capabilities.webgpu ? 'webgpu' : 'wasm'],
    })
  }
  console.timeEnd('sessionCreate')
  console.time('preProcess')

  // 并行加载原始图像和蒙版图像
  const [originalImg, originalMark] = await Promise.all([
    // 如果输入已经是图像元素就直接使用，否则从文件创建
    imageFile instanceof HTMLImageElement
      ? imageFile
      : loadImage(URL.createObjectURL(imageFile)),
    // 加载蒙版图像
    loadImage(maskBase64),
  ])

  // 并行处理图像和蒙版
  const [img, mark] = await Promise.all([
    // 处理原图
    processImage(originalImg),
    // 调整蒙版尺寸与原图匹配，并处理
    processMark(
      await resizeMark(originalMark, originalImg.width, originalImg.height)
    ),
  ])

  // 创建图像张量 - NCHW格式(批次-通道-高度-宽度)
  const imageTensor = new ort.Tensor('uint8', img, [
    1,                  // 批次大小
    3,                  // RGB三通道
    originalImg.height, // 高度
    originalImg.width,  // 宽度
  ])

  // 创建蒙版张量 - NCHW格式
  const maskTensor = new ort.Tensor('uint8', mark, [
    1,                  // 批次大小
    1,                  // 单通道(灰度)
    originalImg.height, // 高度
    originalImg.width,  // 宽度
  ])

  // 准备模型输入
  const Feed: {
    [key: string]: any
  } = {
    [model.inputNames[0]]: imageTensor, // 原始图像
    [model.inputNames[1]]: maskTensor,  // 蒙版图像
  }

  console.timeEnd('preProcess')

  // 运行模型推理
  console.time('run')
  const results = await model.run(Feed)
  console.timeEnd('run')

  // 处理模型输出
  console.time('postProcess')
  // 获取输出张量(修复后的图像)
  const outsTensor = results[model.outputNames[0]]
  // 将CHW格式转换为HWC格式，并转换像素值范围
  const chwToHwcData = postProcess(
    outsTensor.data,
    originalImg.width,
    originalImg.height
  )
  // 创建ImageData对象用于显示
  const imageData = new ImageData(
    new Uint8ClampedArray(chwToHwcData),
    originalImg.width,
    originalImg.height
  )
  console.log(imageData, 'imageData')
  // 转换为Data URL
  const result = imageDataToDataURL(imageData)
  console.timeEnd('postProcess')

  // 返回修复后的图像URL
  return result
}