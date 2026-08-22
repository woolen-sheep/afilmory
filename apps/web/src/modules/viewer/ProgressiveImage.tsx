import { clsxm } from '@afilmory/utils'
import type { WebGLViewportState } from '@afilmory/webgl-viewer'
import { WebGLImageViewer } from '@afilmory/webgl-viewer'
import { AnimatePresence, m } from 'motion/react'
import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { ReactZoomPanPinchRef } from 'react-zoom-pan-pinch'
import { useMediaQuery } from 'usehooks-ts'

import { useShowContextMenu } from '~/atoms/context-menu'
import { SlidingNumber } from '~/components/ui/number/SlidingNumber'
import { isMobileDevice } from '~/lib/device-viewport'
import { canUseWebGL } from '~/lib/feature'
import { HDRBadge } from '~/modules/media/HDRBadge'
import { LivePhotoBadge } from '~/modules/media/LivePhotoBadge'
import type { LivePhotoVideoHandle } from '~/modules/media/LivePhotoVideo'
import { LivePhotoVideo } from '~/modules/media/LivePhotoVideo'

import { ContainedImageFrame } from './ContainedImageFrame'
import { DOMImageViewer } from './DOMImageViewer'
import { getProgressiveImageVisualReady, isThumbnailElementVisuallyReady } from './entry-animation-state'
import {
  createContextMenuItems,
  useImageLoader,
  useLivePhotoControls,
  useProgressiveImageState,
  useScaleIndicator,
  useWebGLLoadingState,
} from './hooks'
import { PhotoRegionsOverlay } from './PhotoRegionsOverlay'
import type { ProgressiveImageProps, WebGLImageViewerRef } from './types'

const loadedThumbnailSrcSet = new Set<string>()

export const ProgressiveImage = ({
  src,
  thumbnailSrc,
  alt,
  width,
  height,
  className,
  onError,
  onProgress,
  onZoomChange,
  onBlobSrcChange,
  onVisualReadyChange,
  disableThumbnailTransition = false,
  enableZoom = true,
  enablePan = true,
  maxZoom = 20,
  minZoom = 1,
  isCurrentImage = false,
  shouldRenderHighRes = true,
  videoSource,
  shouldAutoPlayVideoOnce = false,
  regions = [],
  regionOrientation,
  regionAccentColor,
  activeRegionId,
  showAllRegions = false,
  enableRegionHover = true,
  onActiveRegionChange,
  isHDR = false,
  loadingIndicatorRef,
}: ProgressiveImageProps) => {
  const { t } = useTranslation()

  // State management
  const [state, setState] = useProgressiveImageState()
  const {
    blobSrc,
    highResLoaded,
    error,
    isHighResImageRendered,
    currentScale,
    showScaleIndicator,
    isThumbnailLoaded,
    isLivePhotoPlaying,
  } = state

  const isActiveImage = Boolean(isCurrentImage && shouldRenderHighRes)

  // 判断是否有视频内容（Live Photo 或 Motion Photo）
  const hasVideo = Boolean(videoSource && videoSource.type !== 'none')

  // Refs
  const thumbnailRef = useRef<HTMLImageElement>(null)
  const webglImageViewerRef = useRef<WebGLImageViewerRef | null>(null)
  const webglViewerHostRef = useRef<HTMLDivElement>(null)
  const webglRegionsFrameRef = useRef<HTMLDivElement>(null)
  const domImageViewerRef = useRef<ReactZoomPanPinchRef>(null)
  const livePhotoRef = useRef<LivePhotoVideoHandle | null>(null)
  const [hasWebGLRegionsViewport, setHasWebGLRegionsViewport] = useState(false)
  const hasWebGLRegionsViewportRef = useRef(false)

  const resolvedSrc = useMemo(() => {
    if (src.startsWith('/')) {
      return new URL(src, window.location.origin).toString()
    }
    return src
  }, [src])

  // Hooks
  const imageLoaderManagerRef = useImageLoader(
    resolvedSrc,
    isCurrentImage,
    highResLoaded,
    error,
    onProgress,
    onError,
    onBlobSrcChange,
    loadingIndicatorRef,
    setState.setBlobSrc,
    setState.setHighResLoaded,
    setState.setError,
    setState.setIsHighResImageRendered,
  )

  const { onTransformed, onDOMTransformed } = useScaleIndicator(
    onZoomChange,
    setState.setCurrentScale,
    setState.setShowScaleIndicator,
  )

  const { handleLongPressStart, handleLongPressEnd } = useLivePhotoControls(hasVideo, isLivePhotoPlaying, livePhotoRef)

  const handleWebGLLoadingStateChange = useWebGLLoadingState(loadingIndicatorRef)

  const handleThumbnailLoad = useCallback(() => {
    if (thumbnailSrc) {
      loadedThumbnailSrcSet.add(thumbnailSrc)
    }
    setState.setIsThumbnailLoaded(true)
  }, [setState, thumbnailSrc])

  useLayoutEffect(() => {
    if (!thumbnailSrc) {
      setState.setIsThumbnailLoaded(false)
      return
    }

    const thumbnailElement = thumbnailRef.current
    const isAlreadyLoaded
      = loadedThumbnailSrcSet.has(thumbnailSrc)
        || isThumbnailElementVisuallyReady({
          currentSrc: thumbnailElement?.currentSrc,
          naturalWidth: thumbnailElement?.naturalWidth,
          src: thumbnailElement?.src,
          thumbnailSrc,
        })

    if (isAlreadyLoaded) {
      loadedThumbnailSrcSet.add(thumbnailSrc)
      setState.setIsThumbnailLoaded(true)
      return
    }

    setState.setIsThumbnailLoaded(false)
  }, [setState, thumbnailSrc])

  const showContextMenu = useShowContextMenu()

  const isHDRSupported = useMediaQuery('(dynamic-range: high)')
  // Only use HDR if the browser supports it and the image is HDR
  const shouldUseHDR = isHDR && isHDRSupported
  const hasRegions = regions.length > 0
  const shouldUseDOMViewer = hasVideo || shouldUseHDR
  const shouldRenderThumbnailPhase = Boolean(thumbnailSrc && (!highResLoaded || !blobSrc || !isActiveImage || error))

  const webglPinchConfig = useMemo(
    () => ({
      step: 0.5,
      disabled: !enableZoom,
    }),
    [enableZoom],
  )

  const webglDoubleClickConfig = useMemo(
    () => ({
      step: 2,
      disabled: !enableZoom,
      mode: 'toggle' as const,
      animationTime: 200,
    }),
    [enableZoom],
  )

  const webglPanningConfig = useMemo(
    () => ({
      disabled: !enablePan,
    }),
    [enablePan],
  )

  const isVisualReady = getProgressiveImageVisualReady({
    isHighResImageRendered,
    isThumbnailLoaded,
    thumbnailSrc,
  })

  useLayoutEffect(() => {
    onVisualReadyChange?.(isVisualReady)
  }, [isVisualReady, onVisualReadyChange])

  const handleWebGLViewportChange = useCallback((viewport: WebGLViewportState) => {
    const overlayFrame = webglRegionsFrameRef.current
    if (!overlayFrame) {
      return
    }

    const renderedWidth = viewport.imageWidth * viewport.scale
    const renderedHeight = viewport.imageHeight * viewport.scale
    const left = viewport.containerWidth / 2 + viewport.translateX - renderedWidth / 2
    const top = viewport.containerHeight / 2 + viewport.translateY - renderedHeight / 2

    overlayFrame.style.width = `${renderedWidth}px`
    overlayFrame.style.height = `${renderedHeight}px`
    overlayFrame.style.transform = `translate3d(${left}px, ${top}px, 0)`

    if (!hasWebGLRegionsViewportRef.current) {
      hasWebGLRegionsViewportRef.current = true
      setHasWebGLRegionsViewport(true)
    }
  }, [])

  const getWebGLCanvas = useCallback(() => {
    const canvas = webglViewerHostRef.current?.querySelector('canvas')
    return canvas instanceof HTMLCanvasElement ? canvas : null
  }, [])

  const shouldForwardWebGLEvent = useCallback((target: EventTarget | null, canvas: HTMLCanvasElement) => {
    return target instanceof Node && !canvas.contains(target)
  }, [])

  const dispatchMouseEventToCanvas = useCallback(
    (type: 'mousedown' | 'mousemove' | 'mouseup' | 'dblclick', event: React.MouseEvent<HTMLDivElement>) => {
      const canvas = getWebGLCanvas()
      if (!canvas || !shouldForwardWebGLEvent(event.target, canvas)) {
        return
      }

      event.preventDefault()
      event.stopPropagation()

      canvas.dispatchEvent(
        new MouseEvent(type, {
          bubbles: true,
          cancelable: true,
          button: event.button,
          buttons: event.buttons,
          clientX: event.clientX,
          clientY: event.clientY,
          altKey: event.altKey,
          ctrlKey: event.ctrlKey,
          metaKey: event.metaKey,
          shiftKey: event.shiftKey,
        }),
      )
    },
    [getWebGLCanvas, shouldForwardWebGLEvent],
  )

  const dispatchTouchEventToCanvas = useCallback(
    (type: 'touchstart' | 'touchmove' | 'touchend', event: React.TouchEvent<HTMLDivElement>) => {
      const canvas = getWebGLCanvas()
      if (!canvas || !shouldForwardWebGLEvent(event.target, canvas) || typeof TouchEvent === 'undefined') {
        return
      }

      event.preventDefault()
      event.stopPropagation()

      canvas.dispatchEvent(
        new TouchEvent(type, {
          bubbles: true,
          cancelable: true,
          touches: Array.from(event.nativeEvent.touches),
          targetTouches: Array.from(event.nativeEvent.targetTouches),
          changedTouches: Array.from(event.nativeEvent.changedTouches),
          altKey: event.altKey,
          ctrlKey: event.ctrlKey,
          metaKey: event.metaKey,
          shiftKey: event.shiftKey,
        }),
      )
    },
    [getWebGLCanvas, shouldForwardWebGLEvent],
  )

  const handleWebGLRegionWheel = useCallback(
    (event: React.WheelEvent<HTMLDivElement>) => {
      const canvas = webglViewerHostRef.current?.querySelector('canvas')
      if (!(canvas instanceof HTMLCanvasElement) || !shouldForwardWebGLEvent(event.target, canvas)) {
        return
      }

      event.preventDefault()
      event.stopPropagation()

      canvas.dispatchEvent(
        new WheelEvent('wheel', {
          bubbles: true,
          cancelable: true,
          clientX: event.clientX,
          clientY: event.clientY,
          deltaX: event.deltaX,
          deltaY: event.deltaY,
          deltaZ: event.deltaZ,
          deltaMode: event.deltaMode,
          altKey: event.altKey,
          ctrlKey: event.ctrlKey,
          metaKey: event.metaKey,
          shiftKey: event.shiftKey,
        }),
      )
    },
    [shouldForwardWebGLEvent],
  )

  useLayoutEffect(() => {
    hasWebGLRegionsViewportRef.current = false
    setHasWebGLRegionsViewport(false)

    const overlayFrame = webglRegionsFrameRef.current
    if (!overlayFrame) {
      return
    }

    overlayFrame.style.width = '0px'
    overlayFrame.style.height = '0px'
    overlayFrame.style.transform = 'translate3d(0px, 0px, 0)'
  }, [blobSrc, highResLoaded, isActiveImage, src])

  return (
    <div
      className={clsxm('relative overflow-hidden', className)}
      onMouseDown={handleLongPressStart}
      onMouseUp={handleLongPressEnd}
      onMouseLeave={handleLongPressEnd}
      onTouchStart={handleLongPressStart}
      onTouchEnd={handleLongPressEnd}
    >
      {/* 缩略图 - 在高分辨率图片未加载或加载失败时显示 */}
      {thumbnailSrc && (!isHighResImageRendered || error) && (
        <ContainedImageFrame
          width={width}
          height={height}
          className="absolute inset-0 flex h-full w-full items-center justify-center overflow-visible"
        >
          <img
            ref={thumbnailRef}
            src={thumbnailSrc}
            key={thumbnailSrc}
            alt={alt}
            className={clsxm(
              'block size-full object-contain transition-opacity duration-300',
              disableThumbnailTransition && 'transition-none',
              isThumbnailLoaded ? 'opacity-100' : 'opacity-0',
            )}
            onLoad={handleThumbnailLoad}
          />
          {hasRegions && shouldRenderThumbnailPhase && (
            <PhotoRegionsOverlay
              regions={regions}
              photoWidth={width}
              photoHeight={height}
              orientation={regionOrientation}
              accentColor={regionAccentColor}
              activeRegionId={activeRegionId}
              showAllBoxes={showAllRegions}
              interactive={enableRegionHover}
              onActiveRegionChange={onActiveRegionChange}
            />
          )}
        </ContainedImageFrame>
      )}

      {/* 高分辨率图片 - 只在成功加载且非错误状态时显示 */}
      {highResLoaded && blobSrc && isActiveImage && !error && (
        <div
          className="absolute inset-0 h-full w-full"
          onContextMenu={(e) => {
            const items = createContextMenuItems(blobSrc, alt, t)
            showContextMenu(items, e)
          }}
        >
          {/* LivePhoto/Motion Photo 或 HDR 模式使用 DOMImageViewer */}
          {shouldUseDOMViewer ? (
            <DOMImageViewer
              ref={domImageViewerRef}
              onZoomChange={onDOMTransformed}
              minZoom={minZoom}
              maxZoom={maxZoom}
              enableZoom={enableZoom}
              enablePan={enablePan}
              src={blobSrc}
              alt={alt}
              width={width}
              height={height}
              highResLoaded={highResLoaded}
              onLoad={() => setState.setIsHighResImageRendered(true)}
            >
              {/* LivePhoto/Motion Photo 视频组件作为 children，跟随图片的变换 */}
              {hasVideo && videoSource && imageLoaderManagerRef.current && (
                <LivePhotoVideo
                  ref={livePhotoRef}
                  videoSource={videoSource}
                  imageLoaderManager={imageLoaderManagerRef.current}
                  loadingIndicatorRef={loadingIndicatorRef}
                  isCurrentImage={isCurrentImage}
                  onPlayingChange={setState.setIsLivePhotoPlaying}
                  shouldAutoPlayOnce={shouldAutoPlayVideoOnce}
                />
              )}
              {hasRegions && (
                <PhotoRegionsOverlay
                  regions={regions}
                  photoWidth={width}
                  photoHeight={height}
                  orientation={regionOrientation}
                  accentColor={regionAccentColor}
                  activeRegionId={activeRegionId}
                  showAllBoxes={showAllRegions}
                  interactive={enableRegionHover}
                  onActiveRegionChange={onActiveRegionChange}
                />
              )}
            </DOMImageViewer>
          ) : (
            /* 非 LivePhoto 模式使用 WebGLImageViewer */
            <div
              ref={webglViewerHostRef}
              className="absolute inset-0 h-full w-full"
              onWheelCapture={handleWebGLRegionWheel}
              onMouseDownCapture={event => dispatchMouseEventToCanvas('mousedown', event)}
              onMouseMoveCapture={event => dispatchMouseEventToCanvas('mousemove', event)}
              onMouseUpCapture={event => dispatchMouseEventToCanvas('mouseup', event)}
              onDoubleClickCapture={event => dispatchMouseEventToCanvas('dblclick', event)}
              onTouchStartCapture={event => dispatchTouchEventToCanvas('touchstart', event)}
              onTouchMoveCapture={event => dispatchTouchEventToCanvas('touchmove', event)}
              onTouchEndCapture={event => dispatchTouchEventToCanvas('touchend', event)}
            >
              <WebGLImageViewer
                ref={webglImageViewerRef}
                src={blobSrc}
                className="absolute inset-0 h-full w-full"
                width={width}
                height={height}
                initialScale={1}
                minScale={minZoom}
                maxScale={maxZoom}
                pinch={webglPinchConfig}
                doubleClick={webglDoubleClickConfig}
                panning={webglPanningConfig}
                limitToBounds={true}
                centerOnInit={true}
                smooth={true}
                onZoomChange={onTransformed}
                onViewportChange={handleWebGLViewportChange}
                onLoadingStateChange={handleWebGLLoadingStateChange}
                debug={import.meta.env.DEV && !isMobileDevice}
              />
              {hasRegions && (
                <div className="pointer-events-none absolute inset-0 z-20">
                  <div
                    ref={webglRegionsFrameRef}
                    className={clsxm(
                      'absolute left-0 top-0 overflow-visible transition-opacity duration-200',
                      hasWebGLRegionsViewport ? 'opacity-100' : 'opacity-0',
                    )}
                  >
                    <PhotoRegionsOverlay
                      regions={regions}
                      photoWidth={width}
                      photoHeight={height}
                      orientation={regionOrientation}
                      accentColor={regionAccentColor}
                      activeRegionId={activeRegionId}
                      showAllBoxes={showAllRegions}
                      interactive={enableRegionHover}
                      onActiveRegionChange={onActiveRegionChange}
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {hasVideo && highResLoaded && blobSrc && isActiveImage && !error && (
        <LivePhotoBadge
          livePhotoRef={livePhotoRef}
          isLivePhotoPlaying={isLivePhotoPlaying}
          imageLoaderManagerRef={imageLoaderManagerRef}
        />
      )}

      {shouldUseHDR && highResLoaded && blobSrc && isActiveImage && !error && <HDRBadge />}

      {/* 备用图片（当 WebGL 不可用时） - 只在非错误状态时显示 */}
      {!canUseWebGL && highResLoaded && blobSrc && isActiveImage && !error && (
        <div className="pointer-events-none absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-black/20">
          <i className="i-mingcute-warning-line mb-2 text-4xl" />
          <span className="text-center text-sm text-white">{t('photo.webgl.unavailable')}</span>
        </div>
      )}

      {/* 操作提示 */}
      {!hasVideo && (
        <div className="pointer-events-none absolute bottom-4 left-1/2 z-20 -translate-x-1/2 rounded bg-black/50 px-2 py-1 text-xs text-white opacity-0 duration-200 group-hover:opacity-50">
          {t('photo.zoom.hint')}
        </div>
      )}

      {/* 缩放倍率提示 */}
      <AnimatePresence>
        {showScaleIndicator && (
          <m.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="pointer-events-none absolute bottom-4 left-4 z-20 flex items-center gap-0.5 rounded bg-black/50 px-3 py-1 text-lg text-white tabular-nums"
          >
            <SlidingNumber number={currentScale} decimalPlaces={1} />
            <span>x</span>
          </m.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export type { ProgressiveImageProps } from './types'
