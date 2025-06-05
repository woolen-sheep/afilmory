import { LOD_LEVELS } from './constants'
import type { DebugInfo, MemoryInfo, WebGLImageViewerProps } from './interface'
import {
  createShader,
  FRAGMENT_SHADER_SOURCE,
  VERTEX_SHADER_SOURCE,
} from './shaders'

// 性能监控接口
interface PerformanceInfo {
  fps: number
  frameTime: number
  gpuTime: number
  renderCalls: number
}

// 纹理信息接口
interface TextureInfo {
  texture: WebGLTexture
  width: number
  height: number
  memory: number
  lastUsed: number
}

// 渲染状态 (暂时保留用于未来扩展)
interface _RenderState {
  scale: number
  translateX: number
  translateY: number
  targetLOD: number
  isReady: boolean
}

// Web Worker 消息类型 (暂时保留用于未来扩展)
interface _WorkerMessage {
  type: 'createTexture' | 'textureReady' | 'error' | 'memoryUpdate'
  data: any
}

/**
 * 高性能 WebGL 图像查看器引擎
 * 使用 OffscreenCanvas 双缓冲和内存管理优化
 */
export class WebGLImageViewerEngine {
  private canvas: HTMLCanvasElement
  private gl: WebGLRenderingContext
  private offscreenCanvas: OffscreenCanvas | null = null
  private offscreenGL: WebGLRenderingContext | null = null
  private program!: WebGLProgram
  private offscreenProgram!: WebGLProgram

  // 双缓冲纹理
  private frontTexture: WebGLTexture | null = null
  private backTexture: WebGLTexture | null = null
  private isBackBufferReady = false

  // 图像和状态
  private originalImageSrc = ''
  private imageWidth = 0
  private imageHeight = 0
  private canvasWidth = 0
  private canvasHeight = 0
  private imageLoaded = false

  // 变换状态
  private scale = 1
  private translateX = 0
  private translateY = 0
  private targetScale = 1
  private targetTranslateX = 0
  private targetTranslateY = 0

  // 交互状态
  private isDragging = false
  private lastMouseX = 0
  private lastMouseY = 0
  private lastTouchDistance = 0
  private isOriginalSize = false
  private lastDoubleClickTime = 0
  private lastTouchTime = 0
  private lastTouchX = 0
  private lastTouchY = 0
  private touchTapTimeout: ReturnType<typeof setTimeout> | null = null

  // 动画状态
  private isAnimating = false
  private animationStartTime = 0
  private animationDuration = 300
  private startScale = 1
  private startTranslateX = 0
  private startTranslateY = 0

  // LOD 和纹理管理
  private currentLOD = 0
  private lodTextures = new Map<number, TextureInfo>()
  private texturePool = new Set<WebGLTexture>()
  private maxTextureSize = 0
  private textureMemoryLimit = 0

  // 内存和性能监控
  private memoryInfo: MemoryInfo = {
    usedMemory: 0,
    totalMemory: 0,
    textureMemory: 0,
    pressure: 'low',
  }
  private performanceInfo: PerformanceInfo = {
    fps: 60,
    frameTime: 16.67,
    gpuTime: 0,
    renderCalls: 0,
  }

  // Web Worker 相关
  private textureWorker: Worker | null = null
  private pendingTextureRequests = new Map<
    string,
    (texture: WebGLTexture) => void
  >()

  // 渲染控制
  private renderLoop: number | null = null
  private lastFrameTime = 0
  private frameCount = 0
  private lastFPSUpdate = 0
  private lodUpdateDebounceId: ReturnType<typeof setTimeout> | null = null

  // 配置和回调
  private config: Required<WebGLImageViewerProps>
  private onZoomChange?: (originalScale: number, relativeScale: number) => void
  private onImageCopied?: () => void
  private onDebugUpdate?: React.RefObject<(debugInfo: DebugInfo) => void>

  // 绑定的事件处理器
  private boundHandleMouseDown: (e: MouseEvent) => void
  private boundHandleMouseMove: (e: MouseEvent) => void
  private boundHandleMouseUp: () => void
  private boundHandleWheel: (e: WheelEvent) => void
  private boundHandleDoubleClick: (e: MouseEvent) => void
  private boundHandleTouchStart: (e: TouchEvent) => void
  private boundHandleTouchMove: (e: TouchEvent) => void
  private boundHandleTouchEnd: (e: TouchEvent) => void
  private boundResizeCanvas: () => void

  constructor(
    canvas: HTMLCanvasElement,
    config: Required<WebGLImageViewerProps>,
    onDebugUpdate?: React.RefObject<(debugInfo: any) => void>,
  ) {
    this.canvas = canvas
    this.config = config
    this.onZoomChange = config.onZoomChange
    this.onImageCopied = config.onImageCopied
    this.onDebugUpdate = onDebugUpdate

    // 初始化 WebGL 上下文
    const gl = canvas.getContext('webgl', {
      alpha: true,
      premultipliedAlpha: false,
      antialias: false, // 关闭抗锯齿以提高性能
      powerPreference: 'high-performance',
      failIfMajorPerformanceCaveat: false,
      preserveDrawingBuffer: false, // 不保留绘图缓冲区以节省内存
    })

    if (!gl) {
      throw new Error('WebGL not supported')
    }
    this.gl = gl

    // 获取系统限制
    this.maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE)
    this.calculateMemoryLimits()

    // 绑定事件处理器
    this.boundHandleMouseDown = (e: MouseEvent) => this.handleMouseDown(e)
    this.boundHandleMouseMove = (e: MouseEvent) => this.handleMouseMove(e)
    this.boundHandleMouseUp = () => this.handleMouseUp()
    this.boundHandleWheel = (e: WheelEvent) => this.handleWheel(e)
    this.boundHandleDoubleClick = (e: MouseEvent) => this.handleDoubleClick(e)
    this.boundHandleTouchStart = (e: TouchEvent) => this.handleTouchStart(e)
    this.boundHandleTouchMove = (e: TouchEvent) => this.handleTouchMove(e)
    this.boundHandleTouchEnd = (e: TouchEvent) => this.handleTouchEnd(e)
    this.boundResizeCanvas = () => this.resizeCanvas()

    // 同步初始化关键组件
    this.setupCanvas()
    this.initWebGL()

    // 立即进行一次渲染测试
    gl.clearColor(0.2, 0.2, 0.2, 1) // 灰色背景，确认WebGL工作
    gl.clear(gl.COLOR_BUFFER_BIT)

    this.setupEventListeners()
    this.startRenderLoop()
    this.startMemoryMonitoring()

    // 异步初始化 OffscreenCanvas
    this.setupOffscreenCanvas().catch((error) => {
      console.warn('OffscreenCanvas setup failed:', error)
    })

    console.info('WebGL Image Viewer Engine initialized')
    console.info('Max texture size:', this.maxTextureSize)
    console.info(
      'Texture memory limit:',
      Math.round(this.textureMemoryLimit / 1024 / 1024),
      'MB',
    )
    console.info('Canvas size:', `${this.canvas.width}x${this.canvas.height}`)
    console.info('Canvas CSS size:', `${this.canvasWidth}x${this.canvasHeight}`)
  }

  private async setupOffscreenCanvas() {
    try {
      // 尝试创建 OffscreenCanvas
      if (typeof OffscreenCanvas !== 'undefined') {
        this.offscreenCanvas = new OffscreenCanvas(1024, 1024)
        const offscreenGL = this.offscreenCanvas.getContext('webgl', {
          alpha: true,
          premultipliedAlpha: false,
          antialias: false,
          powerPreference: 'high-performance',
          preserveDrawingBuffer: false,
        })

        if (offscreenGL) {
          this.offscreenGL = offscreenGL as WebGLRenderingContext
          this.initOffscreenWebGL()
          console.info('OffscreenCanvas initialized successfully')
        } else {
          console.warn(
            'Failed to get OffscreenCanvas WebGL context, falling back to main thread',
          )
        }
      } else {
        console.warn(
          'OffscreenCanvas not supported, falling back to main thread',
        )
      }
    } catch (error) {
      console.warn('OffscreenCanvas setup failed:', error)
    }
  }

  private calculateMemoryLimits() {
    // 估算可用内存
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent)
    const devicePixelRatio = window.devicePixelRatio || 1

    // 基于设备类型和屏幕分辨率估算内存限制
    let baseMemoryLimit: number

    if (isMobile) {
      // 移动设备内存限制更严格
      if (devicePixelRatio >= 3) {
        baseMemoryLimit = 256 * 1024 * 1024 // 256MB for high-DPI mobile
      } else {
        baseMemoryLimit = 128 * 1024 * 1024 // 128MB for normal mobile
      }
    } else {
      // 桌面设备
      baseMemoryLimit = 512 * 1024 * 1024 // 512MB for desktop
    }

    // 为纹理分配总内存的 60%
    this.textureMemoryLimit = Math.floor(baseMemoryLimit * 0.6)

    console.info(
      `Device: ${isMobile ? 'Mobile' : 'Desktop'}, DPR: ${devicePixelRatio}`,
    )
    console.info(
      `Memory limit: ${Math.round(this.textureMemoryLimit / 1024 / 1024)}MB`,
    )
  }

  private setupCanvas() {
    this.resizeCanvas()
    window.addEventListener('resize', this.boundResizeCanvas)
  }

  private resizeCanvas() {
    const rect = this.canvas.getBoundingClientRect()
    const devicePixelRatio = window.devicePixelRatio || 1

    this.canvasWidth = rect.width
    this.canvasHeight = rect.height

    // 根据内存压力动态调整像素比
    const effectivePixelRatio = this.getEffectivePixelRatio(devicePixelRatio)

    const actualWidth = Math.round(rect.width * effectivePixelRatio)
    const actualHeight = Math.round(rect.height * effectivePixelRatio)

    // 设置canvas的实际像素尺寸
    this.canvas.width = actualWidth
    this.canvas.height = actualHeight

    // 设置canvas的CSS显示尺寸
    this.canvas.style.width = `${rect.width}px`
    this.canvas.style.height = `${rect.height}px`

    this.gl.viewport(0, 0, actualWidth, actualHeight)

    // 同步 OffscreenCanvas 尺寸
    if (this.offscreenCanvas && this.offscreenGL) {
      this.offscreenCanvas.width = actualWidth
      this.offscreenCanvas.height = actualHeight
      this.offscreenGL.viewport(0, 0, actualWidth, actualHeight)
    }

    if (this.imageLoaded) {
      this.constrainScaleAndPosition()
      this.updateLODIfNeeded()
      this.notifyZoomChange()
    }
  }

  private getEffectivePixelRatio(basePixelRatio: number): number {
    switch (this.memoryInfo.pressure) {
      case 'critical': {
        return Math.min(basePixelRatio, 1)
      } // 最低质量
      case 'high': {
        return Math.min(basePixelRatio, 1.5)
      } // 中等质量
      case 'medium': {
        return Math.min(basePixelRatio, 2)
      } // 较高质量
      default: {
        return basePixelRatio
      } // 最高质量
    }
  }

  private initWebGL() {
    const { gl } = this

    try {
      this.program = this.createShaderProgram(gl)
      this.setupGeometry(gl, this.program)

      // 检查 WebGL 错误
      const error = gl.getError()
      if (error !== gl.NO_ERROR) {
        console.error('WebGL error during initialization:', error)
      } else {
        console.info('WebGL initialized successfully')
      }
    } catch (error) {
      console.error('Failed to initialize WebGL:', error)
      throw error
    }
  }

  private initOffscreenWebGL() {
    if (!this.offscreenGL) return

    this.offscreenProgram = this.createShaderProgram(this.offscreenGL)
    this.setupGeometry(this.offscreenGL, this.offscreenProgram)
  }

  private createShaderProgram(gl: WebGLRenderingContext): WebGLProgram {
    const vertexShader = createShader(
      gl,
      gl.VERTEX_SHADER,
      VERTEX_SHADER_SOURCE,
    )
    const fragmentShader = createShader(
      gl,
      gl.FRAGMENT_SHADER,
      FRAGMENT_SHADER_SOURCE,
    )

    const program = gl.createProgram()!
    gl.attachShader(program, vertexShader)
    gl.attachShader(program, fragmentShader)
    gl.linkProgram(program)

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(
        `Program linking failed: ${gl.getProgramInfoLog(program)}`,
      )
    }

    gl.useProgram(program)
    gl.enable(gl.BLEND)
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)

    return program
  }

  private setupGeometry(gl: WebGLRenderingContext, program: WebGLProgram) {
    const positions = new Float32Array([
      -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1,
    ])
    const texCoords = new Float32Array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0])

    // Position buffer
    const positionBuffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW)

    const positionLocation = gl.getAttribLocation(program, 'a_position')
    gl.enableVertexAttribArray(positionLocation)
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0)

    // Texture coordinate buffer
    const texCoordBuffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW)

    const texCoordLocation = gl.getAttribLocation(program, 'a_texCoord')
    gl.enableVertexAttribArray(texCoordLocation)
    gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, 0, 0)
  }

  async loadImage(url: string) {
    this.originalImageSrc = url

    try {
      console.info('Loading image:', url)
      const image = await this.loadImageAsync(url)
      this.imageWidth = image.width
      this.imageHeight = image.height

      if (this.config.centerOnInit) {
        this.fitImageToScreen()
      } else {
        const fitToScreenScale = this.getFitToScreenScale()
        this.scale = fitToScreenScale * this.config.initialScale
      }

      console.info('Creating initial texture...')
      await this.createInitialTexture(image)
      this.imageLoaded = true

      console.info('Image loaded and texture created successfully')
      console.info(`Image size: ${this.imageWidth}×${this.imageHeight}`)
      console.info(`Canvas size: ${this.canvasWidth}×${this.canvasHeight}`)
      console.info(`Scale: ${this.scale}`)
      console.info(`Front texture:`, this.frontTexture)

      this.updateLODIfNeeded()
      this.notifyZoomChange()
    } catch (error) {
      console.error('Failed to load image:', error)
      throw error
    }
  }

  private loadImageAsync(url: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const image = new Image()
      image.crossOrigin = 'anonymous'
      image.onload = () => resolve(image)
      image.onerror = () => reject(new Error('Failed to load image'))
      image.src = url
    })
  }

  private async createInitialTexture(image: HTMLImageElement) {
    // 创建初始纹理（适应屏幕大小的质量）
    const optimalLOD = this.selectOptimalLOD()
    console.info(`Creating initial texture at LOD ${optimalLOD}`)

    const texture = await this.createLODTextureAsync(optimalLOD, image)
    if (texture) {
      this.currentLOD = optimalLOD
      this.frontTexture = texture
      console.info('Initial texture created successfully')
    } else {
      console.error(
        'Failed to create initial texture, falling back to synchronous method',
      )
      // 降级到同步方法作为备用
      const fallbackTexture = this.createTextureMainThread(optimalLOD, image)
      if (fallbackTexture) {
        this.currentLOD = optimalLOD
        this.frontTexture = fallbackTexture
        console.info('Fallback texture created successfully')
      } else {
        throw new Error(
          'Failed to create texture with both async and sync methods',
        )
      }
    }
  }

  private async createLODTextureAsync(
    lodLevel: number,
    sourceImage?: HTMLImageElement,
  ): Promise<WebGLTexture | null> {
    if (lodLevel < 0 || lodLevel >= LOD_LEVELS.length) return null

    const image =
      sourceImage || (await this.loadImageAsync(this.originalImageSrc))

    try {
      // 优先使用 OffscreenCanvas 异步创建
      if (this.offscreenGL) {
        console.info(`Creating texture ${lodLevel} using OffscreenCanvas`)
        const texture = await this.createTextureOffscreen(lodLevel, image)
        if (texture) {
          console.info(
            `OffscreenCanvas texture ${lodLevel} created successfully`,
          )
          return texture
        }
        console.warn(
          `OffscreenCanvas texture ${lodLevel} creation failed, falling back to main thread`,
        )
      }

      // 降级到主线程，但使用requestIdleCallback来减少阻塞
      console.info(
        `Creating texture ${lodLevel} using main thread with idle callback`,
      )
      return await this.createTextureMainThreadAsync(lodLevel, image)
    } catch (error) {
      console.error(`Failed to create LOD ${lodLevel} texture:`, error)
      return null
    }
  }

  private async createTextureOffscreen(
    lodLevel: number,
    image: HTMLImageElement,
  ): Promise<WebGLTexture | null> {
    if (!this.offscreenGL || !this.offscreenCanvas) return null

    const lodConfig = LOD_LEVELS[lodLevel]

    // 计算目标尺寸
    const targetWidth = Math.max(1, Math.round(image.width * lodConfig.scale))
    const targetHeight = Math.max(1, Math.round(image.height * lodConfig.scale))

    // 应用内存限制
    const { width: finalWidth, height: finalHeight } =
      this.applyMemoryConstraints(targetWidth, targetHeight)

    // 在 OffscreenCanvas 中异步创建纹理
    return new Promise((resolve) => {
      // 使用 MessageChannel 来实现真正的异步处理
      const createTexture = () => {
        try {
          // 创建临时 canvas 进行图像处理
          const tempCanvas = new OffscreenCanvas(finalWidth, finalHeight)
          const ctx = tempCanvas.getContext('2d')!

          // 设置高质量渲染
          ctx.imageSmoothingEnabled = true
          ctx.imageSmoothingQuality = lodConfig.scale >= 1 ? 'high' : 'medium'

          // 绘制图像
          ctx.drawImage(image, 0, 0, finalWidth, finalHeight)

          // 在主线程WebGL上下文中创建纹理（因为不能跨上下文共享纹理）
          const texture = this.gl.createTexture()
          if (!texture) {
            resolve(null)
            return
          }

          this.gl.bindTexture(this.gl.TEXTURE_2D, texture)
          this.setTextureParameters(this.gl, lodConfig.scale >= 1)

          // 从OffscreenCanvas创建ImageBitmap，然后在主线程上传
          tempCanvas
            .convertToBlob()
            .then((blob) => {
              createImageBitmap(blob)
                .then((imageBitmap) => {
                  this.gl.texImage2D(
                    this.gl.TEXTURE_2D,
                    0,
                    this.gl.RGBA,
                    this.gl.RGBA,
                    this.gl.UNSIGNED_BYTE,
                    imageBitmap,
                  )

                  // 检查错误
                  if (this.gl.getError() !== this.gl.NO_ERROR) {
                    this.gl.deleteTexture(texture)
                    resolve(null)
                    return
                  }

                  // 存储纹理信息
                  const memoryUsage = finalWidth * finalHeight * 4 // RGBA
                  this.lodTextures.set(lodLevel, {
                    texture,
                    width: finalWidth,
                    height: finalHeight,
                    memory: memoryUsage,
                    lastUsed: Date.now(),
                  })

                  this.updateMemoryUsage()
                  resolve(texture)
                })
                .catch((error) => {
                  console.error('Error creating ImageBitmap:', error)
                  this.gl.deleteTexture(texture)
                  resolve(null)
                })
            })
            .catch((error) => {
              console.error('Error converting to blob:', error)
              this.gl.deleteTexture(texture)
              resolve(null)
            })
        } catch (error) {
          console.error('Error creating offscreen texture:', error)
          resolve(null)
        }
      }

      // 使用 MessageChannel 来确保异步执行
      if (typeof MessageChannel !== 'undefined') {
        const channel = new MessageChannel()
        channel.port2.onmessage = createTexture
        channel.port1.postMessage(null)
      } else {
        setTimeout(createTexture, 0)
      }
    })
  }

  private async createTextureMainThreadAsync(
    lodLevel: number,
    image: HTMLImageElement,
  ): Promise<WebGLTexture | null> {
    return new Promise((resolve) => {
      const createTexture = () => {
        try {
          const texture = this.createTextureMainThread(lodLevel, image)
          resolve(texture)
        } catch (error) {
          console.error('Error in async main thread texture creation:', error)
          resolve(null)
        }
      }

      // 使用 requestIdleCallback 来减少主线程阻塞
      if (typeof requestIdleCallback !== 'undefined') {
        requestIdleCallback(createTexture, { timeout: 1000 })
      } else {
        setTimeout(createTexture, 0)
      }
    })
  }

  private createTextureMainThread(
    lodLevel: number,
    image: HTMLImageElement,
  ): WebGLTexture | null {
    const lodConfig = LOD_LEVELS[lodLevel]
    const { gl } = this

    const targetWidth = Math.max(1, Math.round(image.width * lodConfig.scale))
    const targetHeight = Math.max(1, Math.round(image.height * lodConfig.scale))

    const { width: finalWidth, height: finalHeight } =
      this.applyMemoryConstraints(targetWidth, targetHeight)

    try {
      // 使用更轻量的方法创建纹理
      const texture = gl.createTexture()
      if (!texture) return null

      gl.bindTexture(gl.TEXTURE_2D, texture)
      this.setTextureParameters(gl, lodConfig.scale >= 1)

      // 直接从图像创建纹理，让 GPU 处理缩放
      if (finalWidth === image.width && finalHeight === image.height) {
        // 原始尺寸，直接上传
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.RGBA,
          gl.RGBA,
          gl.UNSIGNED_BYTE,
          image,
        )
      } else {
        // 需要缩放，使用最小的 canvas 处理
        const canvas = document.createElement('canvas')
        canvas.width = finalWidth
        canvas.height = finalHeight

        const ctx = canvas.getContext('2d')!
        ctx.imageSmoothingEnabled = true
        ctx.imageSmoothingQuality = 'medium' // 平衡质量和性能
        ctx.drawImage(image, 0, 0, finalWidth, finalHeight)

        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.RGBA,
          gl.RGBA,
          gl.UNSIGNED_BYTE,
          canvas,
        )
      }

      if (gl.getError() !== gl.NO_ERROR) {
        gl.deleteTexture(texture)
        return null
      }

      const memoryUsage = finalWidth * finalHeight * 4
      this.lodTextures.set(lodLevel, {
        texture,
        width: finalWidth,
        height: finalHeight,
        memory: memoryUsage,
        lastUsed: Date.now(),
      })

      this.updateMemoryUsage()
      return texture
    } catch (error) {
      console.error('Error creating main thread texture:', error)
      return null
    }
  }

  private applyMemoryConstraints(
    width: number,
    height: number,
  ): { width: number; height: number } {
    // 检查单个纹理是否超出内存限制
    const estimatedMemory = width * height * 4 // RGBA
    const maxSingleTextureMemory = this.textureMemoryLimit * 0.3 // 单个纹理不超过总限制的 30%

    if (estimatedMemory > maxSingleTextureMemory) {
      const scaleFactor = Math.sqrt(maxSingleTextureMemory / estimatedMemory)
      return {
        width: Math.max(1, Math.round(width * scaleFactor)),
        height: Math.max(1, Math.round(height * scaleFactor)),
      }
    }

    // 检查 WebGL 最大纹理尺寸
    const maxSize = this.getEffectiveMaxTextureSize()
    if (width > maxSize || height > maxSize) {
      const aspectRatio = width / height
      if (aspectRatio > 1) {
        return { width: maxSize, height: Math.round(maxSize / aspectRatio) }
      } else {
        return { width: Math.round(maxSize * aspectRatio), height: maxSize }
      }
    }

    return { width, height }
  }

  private getEffectiveMaxTextureSize(): number {
    // 根据内存压力动态调整最大纹理尺寸
    switch (this.memoryInfo.pressure) {
      case 'critical': {
        return Math.min(this.maxTextureSize, 2048)
      }
      case 'high': {
        return Math.min(this.maxTextureSize, 4096)
      }
      case 'medium': {
        return Math.min(this.maxTextureSize, 8192)
      }
      default: {
        return this.maxTextureSize
      }
    }
  }

  private setTextureParameters(gl: WebGLRenderingContext, isHighRes: boolean) {
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

    if (isHighRes) {
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    } else {
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    }
  }

  private selectOptimalLOD(): number {
    if (!this.imageLoaded) return 3 // 默认中等质量

    const fitToScreenScale = this.getFitToScreenScale()
    const relativeScale = this.scale / fitToScreenScale

    // 根据内存压力调整 LOD 选择策略
    const pressureModifier = this.getMemoryPressureModifier()

    for (const [i, lodLevel] of LOD_LEVELS.entries()) {
      const adjustedMaxScale = lodLevel.maxViewportScale * pressureModifier
      if (relativeScale <= adjustedMaxScale) {
        return i
      }
    }

    return LOD_LEVELS.length - 1
  }

  private getMemoryPressureModifier(): number {
    switch (this.memoryInfo.pressure) {
      case 'critical': {
        return 0.5
      } // 强制使用更低质量
      case 'high': {
        return 0.7
      }
      case 'medium': {
        return 0.9
      }
      default: {
        return 1
      }
    }
  }

  private async updateLODIfNeeded() {
    const optimalLOD = this.selectOptimalLOD()

    if (optimalLOD === this.currentLOD) return

    // 检查是否已有目标 LOD 纹理
    let targetTexture = this.lodTextures.get(optimalLOD)?.texture

    if (!targetTexture) {
      // 异步创建新纹理到后台缓冲区
      const newTexture = await this.createLODTextureAsync(optimalLOD)
      targetTexture = newTexture || undefined
    }

    if (targetTexture) {
      // 实现双缓冲切换
      this.swapBuffers(targetTexture, optimalLOD)

      // 预加载相邻 LOD
      this.preloadAdjacentLODs(optimalLOD)
    }
  }

  private swapBuffers(newTexture: WebGLTexture, newLOD: number) {
    // 原子性地切换纹理，避免撕裂
    this.backTexture = newTexture
    this.isBackBufferReady = true

    // 在下一帧切换
    requestAnimationFrame(() => {
      if (this.isBackBufferReady) {
        this.frontTexture = this.backTexture
        this.currentLOD = newLOD
        this.isBackBufferReady = false
        console.info(`Smoothly switched to LOD ${newLOD}`)
      }
    })
  }

  private async preloadAdjacentLODs(currentLOD: number) {
    const preloadTasks: Promise<void>[] = []

    // 预加载下一个更高质量的 LOD
    if (currentLOD < LOD_LEVELS.length - 1) {
      const nextLOD = currentLOD + 1
      if (!this.lodTextures.has(nextLOD)) {
        preloadTasks.push(this.createLODTextureAsync(nextLOD).then(() => {}))
      }
    }

    // 预加载下一个更低质量的 LOD
    if (currentLOD > 0) {
      const prevLOD = currentLOD - 1
      if (!this.lodTextures.has(prevLOD)) {
        preloadTasks.push(this.createLODTextureAsync(prevLOD).then(() => {}))
      }
    }

    // 使用空闲时间进行预加载
    if (preloadTasks.length > 0) {
      const runPreload = () => {
        Promise.all(preloadTasks).catch(() => {
          // 忽略预加载错误
        })
      }

      if (typeof requestIdleCallback !== 'undefined') {
        requestIdleCallback(runPreload, { timeout: 1000 })
      } else {
        setTimeout(runPreload, 100)
      }
    }
  }

  private startMemoryMonitoring() {
    const updateMemoryInfo = () => {
      this.updateMemoryUsage()
      this.checkMemoryPressure()

      // 内存压力过高时进行清理
      if (this.memoryInfo.pressure === 'critical') {
        this.emergencyCleanup()
      } else if (this.memoryInfo.pressure === 'high') {
        this.cleanupOldTextures()
      }
    }

    // 每秒更新一次内存信息
    setInterval(updateMemoryInfo, 1000)
    updateMemoryInfo() // 立即执行一次
  }

  private updateMemoryUsage() {
    let textureMemory = 0

    for (const textureInfo of this.lodTextures.values()) {
      textureMemory += textureInfo.memory
    }

    this.memoryInfo.textureMemory = textureMemory
    this.memoryInfo.usedMemory = textureMemory // 简化估算

    // 尝试获取更准确的内存信息
    if ('memory' in performance) {
      const memInfo = (performance as any).memory
      if (memInfo) {
        this.memoryInfo.usedMemory = memInfo.usedJSHeapSize
        this.memoryInfo.totalMemory = memInfo.totalJSHeapSize
      }
    }
  }

  private checkMemoryPressure() {
    const usageRatio = this.memoryInfo.textureMemory / this.textureMemoryLimit

    if (usageRatio > 0.9) {
      this.memoryInfo.pressure = 'critical'
    } else if (usageRatio > 0.7) {
      this.memoryInfo.pressure = 'high'
    } else if (usageRatio > 0.5) {
      this.memoryInfo.pressure = 'medium'
    } else {
      this.memoryInfo.pressure = 'low'
    }
  }

  private emergencyCleanup() {
    console.warn('Emergency memory cleanup triggered')

    // 只保留当前 LOD 和相邻的一个 LOD
    const toKeep = new Set([
      this.currentLOD,
      this.currentLOD - 1,
      this.currentLOD + 1,
    ])

    for (const [lodLevel, textureInfo] of this.lodTextures) {
      if (!toKeep.has(lodLevel)) {
        this.gl.deleteTexture(textureInfo.texture)
        this.lodTextures.delete(lodLevel)
      }
    }

    // 强制垃圾回收（如果支持）
    if ('gc' in window) {
      ;(window as any).gc()
    }
  }

  private cleanupOldTextures() {
    const now = Date.now()
    const maxAge = 30000 // 30秒

    for (const [lodLevel, textureInfo] of this.lodTextures) {
      // 不清理当前正在使用的纹理
      if (lodLevel === this.currentLOD) continue

      if (now - textureInfo.lastUsed > maxAge) {
        this.gl.deleteTexture(textureInfo.texture)
        this.lodTextures.delete(lodLevel)
      }
    }
  }

  private startRenderLoop() {
    // 初始化帧时间
    this.lastFrameTime = performance.now()
    this.lastFPSUpdate = this.lastFrameTime

    const render = (currentTime: number) => {
      // 计算帧时间
      const deltaTime = currentTime - this.lastFrameTime
      this.lastFrameTime = currentTime

      // 更新性能统计
      this.updatePerformanceStats(deltaTime)

      // 执行动画
      if (this.isAnimating) {
        this.updateAnimation(currentTime)
      }

      // 渲染帧
      this.renderFrame()

      // 更新调试信息
      if (this.config.debug && this.onDebugUpdate?.current) {
        this.updateDebugInfo()
      }

      // 继续渲染循环
      this.renderLoop = requestAnimationFrame(render)
    }

    this.renderLoop = requestAnimationFrame(render)
  }

  private updatePerformanceStats(deltaTime: number) {
    this.frameCount++
    this.performanceInfo.frameTime = deltaTime

    // 每秒更新一次 FPS
    if (this.lastFrameTime - this.lastFPSUpdate >= 1000) {
      this.performanceInfo.fps = Math.round(
        (this.frameCount * 1000) / (this.lastFrameTime - this.lastFPSUpdate),
      )
      this.frameCount = 0
      this.lastFPSUpdate = this.lastFrameTime
    }
  }

  private updateAnimation(currentTime: number) {
    const elapsed = currentTime - this.animationStartTime
    const progress = Math.min(elapsed / this.animationDuration, 1)
    const easedProgress = this.config.smooth
      ? this.easeOutQuart(progress)
      : progress

    // 插值动画参数
    this.scale =
      this.startScale + (this.targetScale - this.startScale) * easedProgress
    this.translateX =
      this.startTranslateX +
      (this.targetTranslateX - this.startTranslateX) * easedProgress
    this.translateY =
      this.startTranslateY +
      (this.targetTranslateY - this.startTranslateY) * easedProgress

    if (progress >= 1) {
      // 动画完成
      this.isAnimating = false
      this.scale = this.targetScale
      this.translateX = this.targetTranslateX
      this.translateY = this.targetTranslateY

      // 动画完成后更新 LOD
      this.updateLODIfNeeded()
    }

    this.notifyZoomChange()
  }

  private renderFrame() {
    const { gl } = this

    // 设置视口
    gl.viewport(0, 0, this.canvas.width, this.canvas.height)

    // 清除画布
    gl.clearColor(0, 0, 0, 0)
    gl.clear(gl.COLOR_BUFFER_BIT)

    if (!this.frontTexture || !this.imageLoaded) {
      // 渲染一个调试背景以确认渲染循环在工作
      if (this.config.debug) {
        gl.clearColor(0.1, 0.1, 0.1, 1) // 深灰色背景
        gl.clear(gl.COLOR_BUFFER_BIT)
        console.info(
          'Rendering debug background - frontTexture:',
          !!this.frontTexture,
          'imageLoaded:',
          this.imageLoaded,
        )
      }
      return
    }

    if (this.config.debug) {
      console.info('Rendering frame:', {
        canvasSize: `${this.canvas.width}x${this.canvas.height}`,
        imageSize: `${this.imageWidth}x${this.imageHeight}`,
        scale: this.scale,
        translate: `${this.translateX}, ${this.translateY}`,
        frontTexture: !!this.frontTexture,
      })
    }

    gl.useProgram(this.program)

    // 设置变换矩阵
    const matrixLocation = gl.getUniformLocation(this.program, 'u_matrix')
    const matrix = this.createTransformMatrix()
    gl.uniformMatrix3fv(matrixLocation, false, matrix)

    if (this.config.debug) {
      console.info('Transform matrix:', Array.from(matrix))
    }

    // 绑定纹理
    const imageLocation = gl.getUniformLocation(this.program, 'u_image')
    gl.uniform1i(imageLocation, 0)

    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.frontTexture)

    // 绘制
    gl.drawArrays(gl.TRIANGLES, 0, 6)

    // 检查 WebGL 错误
    const error = gl.getError()
    if (error !== gl.NO_ERROR && this.config.debug) {
      console.error('WebGL error in renderFrame:', error)
    }

    this.performanceInfo.renderCalls++
  }

  private createTransformMatrix(): Float32Array {
    const scaleX = (this.imageWidth * this.scale) / this.canvasWidth
    const scaleY = (this.imageHeight * this.scale) / this.canvasHeight
    const translateX = (this.translateX * 2) / this.canvasWidth
    const translateY = -(this.translateY * 2) / this.canvasHeight

    return new Float32Array([
      scaleX,
      0,
      0,
      0,
      scaleY,
      0,
      translateX,
      translateY,
      1,
    ])
  }

  // 缓动函数
  private easeOutQuart(t: number): number {
    return 1 - Math.pow(1 - t, 4)
  }

  private fitImageToScreen() {
    const scaleX = this.canvasWidth / this.imageWidth
    const scaleY = this.canvasHeight / this.imageHeight
    const fitToScreenScale = Math.min(scaleX, scaleY)

    this.scale = fitToScreenScale * this.config.initialScale
    this.translateX = 0
    this.translateY = 0
    this.isOriginalSize = false
  }

  private startAnimation(
    targetScale: number,
    targetTranslateX: number,
    targetTranslateY: number,
    animationTime?: number,
  ) {
    this.isAnimating = true
    this.animationStartTime = performance.now()
    this.animationDuration = animationTime || (this.config.smooth ? 300 : 0)

    this.startScale = this.scale
    this.targetScale = targetScale
    this.startTranslateX = this.translateX
    this.startTranslateY = this.translateY

    // 应用约束到目标位置
    const tempScale = this.scale
    const tempTranslateX = this.translateX
    const tempTranslateY = this.translateY

    this.scale = targetScale
    this.translateX = targetTranslateX
    this.translateY = targetTranslateY
    this.constrainImagePosition()

    this.targetTranslateX = this.translateX
    this.targetTranslateY = this.translateY

    // 恢复当前状态
    this.scale = tempScale
    this.translateX = tempTranslateX
    this.translateY = tempTranslateY
  }

  private updateDebugInfo() {
    if (!this.onDebugUpdate?.current) return

    const fitToScreenScale = this.getFitToScreenScale()
    const relativeScale = this.scale / fitToScreenScale

    this.onDebugUpdate.current({
      scale: this.scale,
      relativeScale,
      translateX: this.translateX,
      translateY: this.translateY,
      currentLOD: this.currentLOD,
      lodLevels: LOD_LEVELS.length,
      canvasSize: { width: this.canvasWidth, height: this.canvasHeight },
      imageSize: { width: this.imageWidth, height: this.imageHeight },
      fitToScreenScale,
      effectiveMaxScale: Math.max(fitToScreenScale * this.config.maxScale, 1),
      originalSizeScale: 1,
      renderCount: this.performanceInfo.renderCalls,
      maxTextureSize: this.maxTextureSize,
      userMaxScale: this.config.maxScale,
      memoryInfo: { ...this.memoryInfo },
    })
  }

  private getFitToScreenScale(): number {
    const scaleX = this.canvasWidth / this.imageWidth
    const scaleY = this.canvasHeight / this.imageHeight
    return Math.min(scaleX, scaleY)
  }

  private constrainImagePosition() {
    if (!this.config.limitToBounds) return

    const fitScale = this.getFitToScreenScale()

    // If current scale is less than or equal to fit-to-screen scale, center the image
    if (this.scale <= fitScale) {
      this.translateX = 0
      this.translateY = 0
      return
    }

    // Otherwise, constrain the image within reasonable bounds
    const scaledWidth = this.imageWidth * this.scale
    const scaledHeight = this.imageHeight * this.scale

    // Calculate the maximum allowed translation to keep image edges within viewport
    const maxTranslateX = Math.max(0, (scaledWidth - this.canvasWidth) / 2)
    const maxTranslateY = Math.max(0, (scaledHeight - this.canvasHeight) / 2)

    // Constrain translation
    this.translateX = Math.max(
      -maxTranslateX,
      Math.min(maxTranslateX, this.translateX),
    )
    this.translateY = Math.max(
      -maxTranslateY,
      Math.min(maxTranslateY, this.translateY),
    )
  }

  private constrainScaleAndPosition() {
    // 首先约束缩放倍数
    const fitToScreenScale = this.getFitToScreenScale()
    const absoluteMinScale = fitToScreenScale * this.config.minScale

    // 计算原图1x尺寸对应的绝对缩放值
    const originalSizeScale = 1 // 原图1x尺寸

    // 确保maxScale不会阻止用户查看原图1x尺寸
    const userMaxScale = fitToScreenScale * this.config.maxScale
    const effectiveMaxScale = Math.max(userMaxScale, originalSizeScale)

    // 如果当前缩放超出范围，调整到合理范围内
    if (this.scale < absoluteMinScale) {
      this.scale = absoluteMinScale
    } else if (this.scale > effectiveMaxScale) {
      this.scale = effectiveMaxScale
    }

    // 然后约束位置
    this.constrainImagePosition()
  }

  private notifyZoomChange() {
    if (this.onZoomChange) {
      // 原图缩放比例（相对于图片原始大小）
      const originalScale = this.scale

      // 相对于页面适应大小的缩放比例
      const fitToScreenScale = this.getFitToScreenScale()
      const relativeScale = this.scale / fitToScreenScale

      this.onZoomChange(originalScale, relativeScale)
    }
  }

  private setupEventListeners() {
    // Mouse events
    this.canvas.addEventListener('mousedown', this.boundHandleMouseDown)
    this.canvas.addEventListener('mousemove', this.boundHandleMouseMove)
    this.canvas.addEventListener('mouseup', this.boundHandleMouseUp)
    this.canvas.addEventListener('wheel', this.boundHandleWheel)
    this.canvas.addEventListener('dblclick', this.boundHandleDoubleClick)

    // Touch events
    this.canvas.addEventListener('touchstart', this.boundHandleTouchStart)
    this.canvas.addEventListener('touchmove', this.boundHandleTouchMove)
    this.canvas.addEventListener('touchend', this.boundHandleTouchEnd)
  }

  private removeEventListeners() {
    this.canvas.removeEventListener('mousedown', this.boundHandleMouseDown)
    this.canvas.removeEventListener('mousemove', this.boundHandleMouseMove)
    this.canvas.removeEventListener('mouseup', this.boundHandleMouseUp)
    this.canvas.removeEventListener('wheel', this.boundHandleWheel)
    this.canvas.removeEventListener('dblclick', this.boundHandleDoubleClick)
    this.canvas.removeEventListener('touchstart', this.boundHandleTouchStart)
    this.canvas.removeEventListener('touchmove', this.boundHandleTouchMove)
    this.canvas.removeEventListener('touchend', this.boundHandleTouchEnd)
  }

  private handleMouseDown(e: MouseEvent) {
    if (this.isAnimating || this.config.panning.disabled) return

    // Stop any ongoing animation when user starts interacting
    this.isAnimating = false

    this.isDragging = true
    this.lastMouseX = e.clientX
    this.lastMouseY = e.clientY
  }

  private handleMouseMove(e: MouseEvent) {
    if (!this.isDragging || this.config.panning.disabled) return

    const deltaX = e.clientX - this.lastMouseX
    const deltaY = e.clientY - this.lastMouseY

    this.translateX += deltaX
    this.translateY += deltaY

    this.lastMouseX = e.clientX
    this.lastMouseY = e.clientY

    this.constrainImagePosition()
  }

  private handleMouseUp() {
    this.isDragging = false
  }

  private handleWheel(e: WheelEvent) {
    e.preventDefault()

    if (this.isAnimating || this.config.wheel.wheelDisabled) return

    const rect = this.canvas.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const mouseY = e.clientY - rect.top

    const scaleFactor =
      e.deltaY > 0 ? 1 - this.config.wheel.step : 1 + this.config.wheel.step
    this.zoomAt(mouseX, mouseY, scaleFactor)
  }

  private handleDoubleClick(e: MouseEvent) {
    e.preventDefault()

    if (this.config.doubleClick.disabled) return

    const now = Date.now()
    if (now - this.lastDoubleClickTime < 300) return
    this.lastDoubleClickTime = now

    const rect = this.canvas.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const mouseY = e.clientY - rect.top

    this.performDoubleClickAction(mouseX, mouseY)
  }

  private handleTouchDoubleTap(clientX: number, clientY: number) {
    if (this.config.doubleClick.disabled) return

    const rect = this.canvas.getBoundingClientRect()
    const touchX = clientX - rect.left
    const touchY = clientY - rect.top

    this.performDoubleClickAction(touchX, touchY)
  }

  private performDoubleClickAction(x: number, y: number) {
    // Stop any ongoing animation
    this.isAnimating = false

    if (this.config.doubleClick.mode === 'toggle') {
      const fitToScreenScale = this.getFitToScreenScale()
      const absoluteMinScale = fitToScreenScale * this.config.minScale

      // 计算原图1x尺寸对应的绝对缩放值
      const originalSizeScale = 1 // 原图1x尺寸

      // 确保maxScale不会阻止用户查看原图1x尺寸
      const userMaxScale = fitToScreenScale * this.config.maxScale
      const effectiveMaxScale = Math.max(userMaxScale, originalSizeScale)

      if (this.isOriginalSize) {
        // Animate to fit-to-screen 1x (适应页面大小) with click position as center
        const targetScale = Math.max(
          absoluteMinScale,
          Math.min(effectiveMaxScale, fitToScreenScale),
        )

        // Calculate zoom point relative to current transform
        const zoomX = (x - this.canvasWidth / 2 - this.translateX) / this.scale
        const zoomY = (y - this.canvasHeight / 2 - this.translateY) / this.scale

        // Calculate target translation after zoom
        const targetTranslateX = x - this.canvasWidth / 2 - zoomX * targetScale
        const targetTranslateY = y - this.canvasHeight / 2 - zoomY * targetScale

        this.startAnimation(
          targetScale,
          targetTranslateX,
          targetTranslateY,
          this.config.doubleClick.animationTime,
        )
        this.isOriginalSize = false
      } else {
        // Animate to original size 1x (原图原始大小) with click position as center
        // 确保能够缩放到原图1x尺寸，即使超出用户设置的maxScale
        const targetScale = Math.max(
          absoluteMinScale,
          Math.min(effectiveMaxScale, originalSizeScale),
        ) // 1x = 原图原始大小

        // Calculate zoom point relative to current transform
        const zoomX = (x - this.canvasWidth / 2 - this.translateX) / this.scale
        const zoomY = (y - this.canvasHeight / 2 - this.translateY) / this.scale

        // Calculate target translation after zoom
        const targetTranslateX = x - this.canvasWidth / 2 - zoomX * targetScale
        const targetTranslateY = y - this.canvasHeight / 2 - zoomY * targetScale

        this.startAnimation(
          targetScale,
          targetTranslateX,
          targetTranslateY,
          this.config.doubleClick.animationTime,
        )
        this.isOriginalSize = true
      }
    } else {
      // Zoom mode
      this.zoomAt(x, y, this.config.doubleClick.step)
    }
  }

  private handleTouchStart(e: TouchEvent) {
    e.preventDefault()

    if (this.isAnimating) return

    if (e.touches.length === 1 && !this.config.panning.disabled) {
      const touch = e.touches[0]
      const now = Date.now()

      // Check for double-tap
      if (
        !this.config.doubleClick.disabled &&
        now - this.lastTouchTime < 300 &&
        Math.abs(touch.clientX - this.lastTouchX) < 50 &&
        Math.abs(touch.clientY - this.lastTouchY) < 50
      ) {
        // Double-tap detected
        this.handleTouchDoubleTap(touch.clientX, touch.clientY)
        this.lastTouchTime = 0 // Reset to prevent triple-tap
        return
      }

      // Single touch - prepare for potential drag or single tap
      this.isDragging = true
      this.lastMouseX = touch.clientX
      this.lastMouseY = touch.clientY
      this.lastTouchTime = now
      this.lastTouchX = touch.clientX
      this.lastTouchY = touch.clientY
    } else if (e.touches.length === 2 && !this.config.pinch.disabled) {
      this.isDragging = false
      const touch1 = e.touches[0]
      const touch2 = e.touches[1]
      this.lastTouchDistance = Math.sqrt(
        Math.pow(touch2.clientX - touch1.clientX, 2) +
          Math.pow(touch2.clientY - touch1.clientY, 2),
      )
    }
  }

  private handleTouchMove(e: TouchEvent) {
    e.preventDefault()

    if (
      e.touches.length === 1 &&
      this.isDragging &&
      !this.config.panning.disabled
    ) {
      const deltaX = e.touches[0].clientX - this.lastMouseX
      const deltaY = e.touches[0].clientY - this.lastMouseY

      this.translateX += deltaX
      this.translateY += deltaY

      this.lastMouseX = e.touches[0].clientX
      this.lastMouseY = e.touches[0].clientY

      this.constrainImagePosition()
    } else if (e.touches.length === 2 && !this.config.pinch.disabled) {
      const touch1 = e.touches[0]
      const touch2 = e.touches[1]
      const distance = Math.sqrt(
        Math.pow(touch2.clientX - touch1.clientX, 2) +
          Math.pow(touch2.clientY - touch1.clientY, 2),
      )

      if (this.lastTouchDistance > 0) {
        const scaleFactor = distance / this.lastTouchDistance
        const centerX = (touch1.clientX + touch2.clientX) / 2
        const centerY = (touch1.clientY + touch2.clientY) / 2

        const rect = this.canvas.getBoundingClientRect()
        this.zoomAt(centerX - rect.left, centerY - rect.top, scaleFactor)
      }

      this.lastTouchDistance = distance
    }
  }

  private handleTouchEnd(_e: TouchEvent) {
    this.isDragging = false
    this.lastTouchDistance = 0

    // Clear any pending touch tap timeout
    if (this.touchTapTimeout) {
      clearTimeout(this.touchTapTimeout)
      this.touchTapTimeout = null
    }
  }

  private zoomAt(x: number, y: number, scaleFactor: number, animated = false) {
    const newScale = this.scale * scaleFactor
    const fitToScreenScale = this.getFitToScreenScale()

    // 将相对缩放比例转换为绝对缩放比例进行限制
    const absoluteMinScale = fitToScreenScale * this.config.minScale

    // 计算原图 1x 尺寸对应的绝对缩放值
    const originalSizeScale = 1 // 原图 1x 尺寸

    // 确保 maxScale 不会阻止用户查看原图 1x 尺寸
    const userMaxScale = fitToScreenScale * this.config.maxScale
    const effectiveMaxScale = Math.max(userMaxScale, originalSizeScale)

    // Limit zoom
    if (newScale < absoluteMinScale || newScale > effectiveMaxScale) return

    if (animated && this.config.smooth) {
      // Calculate zoom point relative to current transform
      const zoomX = (x - this.canvasWidth / 2 - this.translateX) / this.scale
      const zoomY = (y - this.canvasHeight / 2 - this.translateY) / this.scale

      // Calculate target translation after zoom
      const targetTranslateX = x - this.canvasWidth / 2 - zoomX * newScale
      const targetTranslateY = y - this.canvasHeight / 2 - zoomY * newScale

      this.startAnimation(newScale, targetTranslateX, targetTranslateY)
    } else {
      // Calculate zoom point relative to current transform
      const zoomX = (x - this.canvasWidth / 2 - this.translateX) / this.scale
      const zoomY = (y - this.canvasHeight / 2 - this.translateY) / this.scale

      this.scale = newScale

      // Adjust translation to keep zoom point fixed
      this.translateX = x - this.canvasWidth / 2 - zoomX * this.scale
      this.translateY = y - this.canvasHeight / 2 - zoomY * this.scale

      this.constrainImagePosition()
      this.notifyZoomChange()
      this.updateLODIfNeeded()
    }
  }

  async copyOriginalImageToClipboard() {
    try {
      // 获取原始图片
      const response = await fetch(this.originalImageSrc)
      const blob = await response.blob()

      // 检查浏览器是否支持剪贴板 API
      if (!navigator.clipboard || !navigator.clipboard.write) {
        console.warn('Clipboard API not supported')
        return
      }

      // 创建 ClipboardItem 并写入剪贴板
      const clipboardItem = new ClipboardItem({
        [blob.type]: blob,
      })

      await navigator.clipboard.write([clipboardItem])
      console.info('Original image copied to clipboard')
      if (this.onImageCopied) {
        this.onImageCopied()
      }
    } catch (error) {
      console.error('Failed to copy image to clipboard:', error)
    }
  }

  // Public methods
  public zoomIn(animated = false) {
    const centerX = this.canvasWidth / 2
    const centerY = this.canvasHeight / 2
    this.zoomAt(centerX, centerY, 1 + this.config.wheel.step, animated)
  }

  public zoomOut(animated = false) {
    const centerX = this.canvasWidth / 2
    const centerY = this.canvasHeight / 2
    this.zoomAt(centerX, centerY, 1 - this.config.wheel.step, animated)
  }

  public resetView() {
    const fitToScreenScale = this.getFitToScreenScale()
    const targetScale = fitToScreenScale * this.config.initialScale
    this.startAnimation(targetScale, 0, 0)
  }

  public getScale(): number {
    return this.scale
  }

  public destroy() {
    this.removeEventListeners()
    window.removeEventListener('resize', this.boundResizeCanvas)

    // 清理节流相关的资源
    if (this.renderLoop !== null) {
      cancelAnimationFrame(this.renderLoop)
      this.renderLoop = null
    }

    // 清理 LOD 更新防抖相关的资源
    if (this.lodUpdateDebounceId !== null) {
      clearTimeout(this.lodUpdateDebounceId)
      this.lodUpdateDebounceId = null
    }

    // 清理触摸双击相关的资源
    if (this.touchTapTimeout !== null) {
      clearTimeout(this.touchTapTimeout)
      this.touchTapTimeout = null
    }

    // 清理 WebGL 资源
    this.cleanupWebGLResources()
  }

  private cleanupWebGLResources() {
    const { gl } = this

    // 删除所有纹理
    for (const textureInfo of this.lodTextures.values()) {
      gl.deleteTexture(textureInfo.texture)
    }
    this.lodTextures.clear()

    // 清理纹理池
    for (const texture of this.texturePool) {
      gl.deleteTexture(texture)
    }
    this.texturePool.clear()

    // 清理主纹理引用
    this.frontTexture = null
    this.backTexture = null

    // 删除着色器程序
    if (this.program) {
      gl.deleteProgram(this.program)
    }

    // 清理 OffscreenGL 资源
    if (this.offscreenGL && this.offscreenProgram) {
      this.offscreenGL.deleteProgram(this.offscreenProgram)
    }
  }

  // 添加公共方法用于获取内存和性能信息
  public getMemoryInfo(): MemoryInfo {
    return { ...this.memoryInfo }
  }

  public getPerformanceInfo(): PerformanceInfo {
    return { ...this.performanceInfo }
  }
}
