/**
 * WebGL图片查看器调试信息组件
 *
 * 该组件用于显示WebGL图片查看器的实时调试信息，
 * 包括缩放比例、位置、LOD级别、性能指标等。
 */

import * as React from 'react'
import { useCallback, useImperativeHandle, useState } from 'react'

import type { DebugInfo } from './interface'

/**
 * 调试信息组件的引用接口
 */
export interface DebugInfoRef {
  /** 更新调试信息的方法 */
  updateDebugInfo: (debugInfo: DebugInfo) => void
}

/**
 * 调试信息组件的属性接口
 */
interface DebugInfoProps {
  /** 组件引用 */
  ref: React.Ref<DebugInfoRef>
}

/**
 * 调试信息显示组件
 *
 * 在开发模式下显示WebGL图片查看器的详细状态信息，
 * 帮助开发者诊断性能问题和调试功能。
 *
 * @param props 组件属性
 * @returns JSX元素
 */
const DebugInfoComponent = ({ ref }: DebugInfoProps) => {
  // 调试信息状态，包含所有需要显示的调试数据
  const [debugInfo, setDebugInfo] = useState<DebugInfo>({
    scale: 1,
    relativeScale: 1,
    translateX: 0,
    translateY: 0,
    currentLOD: 0,
    lodLevels: 0,
    canvasSize: { width: 0, height: 0 },
    imageSize: { width: 0, height: 0 },
    fitToScreenScale: 1,
    userMaxScale: 1,
    effectiveMaxScale: 1,
    originalSizeScale: 1,
    renderCount: 0,
    maxTextureSize: 0,
    memoryInfo: {
      usedMemory: 0,
      totalMemory: 0,
      textureMemory: 0,
      pressure: 'low',
    },
  })

  // 暴露更新调试信息的方法给父组件
  useImperativeHandle(
    ref,
    useCallback(
      () => ({
        updateDebugInfo: (debugInfo: DebugInfo) => {
          setDebugInfo(debugInfo)
        },
      }),
      [],
    ),
  )

  return (
    <div
      style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        background: 'rgba(0, 0, 0, 0.8)',
        color: 'white',
        padding: '10px',
        borderRadius: '4px',
        fontSize: '12px',
        fontFamily: 'monospace',
        lineHeight: '1.4',
        pointerEvents: 'none',
        zIndex: 1000,
        minWidth: '200px',
      }}
    >
      {/* 调试面板标题 */}
      <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>
        WebGL Debug Info
      </div>

      {/* 缩放相关信息 */}
      <div>Scale: {debugInfo.scale.toFixed(3)}</div>
      <div>Relative Scale: {debugInfo.relativeScale.toFixed(3)}</div>

      {/* 位置信息 */}
      <div>
        Translate: ({debugInfo.translateX.toFixed(1)},{' '}
        {debugInfo.translateY.toFixed(1)})
      </div>

      {/* Canvas和设备信息 */}
      <div>
        Canvas Size: {debugInfo.canvasSize.width}×{debugInfo.canvasSize.height}
      </div>
      <div>Device Pixel Ratio: {window.devicePixelRatio || 1}</div>

      {/* 图像信息 */}
      <div>
        Image: {debugInfo.imageSize.width}×{debugInfo.imageSize.height}
      </div>

      {/* LOD信息 */}
      <div>
        Current LOD: {debugInfo.currentLOD} / {debugInfo.lodLevels - 1}
      </div>
      <div>Max Texture Size: {debugInfo.maxTextureSize}</div>

      {/* 缩放限制信息 */}
      <div>Fit Scale: {debugInfo.fitToScreenScale.toFixed(3)}</div>
      <div>User Max Scale: {debugInfo.userMaxScale.toFixed(3)}</div>
      <div>Effective Max Scale: {debugInfo.effectiveMaxScale.toFixed(3)}</div>
      <div>Original Size Scale: {debugInfo.originalSizeScale.toFixed(3)}</div>

      {/* 内存信息 */}
      <div>
        Memory Usage: {debugInfo.memoryInfo.usedMemory.toFixed(0)} /{' '}
        {debugInfo.memoryInfo.totalMemory.toFixed(0)} bytes
      </div>
      <div>
        Texture Memory: {debugInfo.memoryInfo.textureMemory.toFixed(0)} bytes
      </div>
      <div>Pressure: {debugInfo.memoryInfo.pressure}</div>
    </div>
  )
}

// 设置显示名称用于React DevTools
DebugInfoComponent.displayName = 'DebugInfo'

// 导出为默认和命名导出
export default DebugInfoComponent
export { DebugInfoComponent as DebugInfo }
