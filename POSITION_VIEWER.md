# Position Viewer Configuration

This document describes how to configure a custom position viewer for GPS locations in photos.

## Overview

By default, Afilmory uses AMap (高德地图) to display GPS coordinates from photos. You can now configure a custom position viewer URL template to use different map services.

## Configuration

Add a `positionViewer` field to your `config.json`:

```json
{
  "positionViewer": "https://uri.amap.com/marker?position={longitude},{latitude}&name={name}"
}
```

## Template Variables

The URL template supports the following variables:

- `{longitude}`: GPS longitude coordinate (e.g., `118.131694`)
- `{latitude}`: GPS latitude coordinate (e.g., `24.502188`) 
- `{name}`: Location name/label (defaults to "拍摄位置")

## Examples

### AMap (Default)
```json
{
  "positionViewer": "https://uri.amap.com/marker?position={longitude},{latitude}&name={name}"
}
```

### OpenStreetMap
```json
{
  "positionViewer": "https://www.openstreetmap.org/?mlat={latitude}&mlon={longitude}&zoom=15"
}
```

### Google Maps
```json
{
  "positionViewer": "https://maps.google.com/?q={latitude},{longitude}"
}
```

### MapLibre with OpenFreeMap (Clean alternative)
```json
{
  "positionViewer": "https://enter-tainer.github.io/pinpoint/?position={longitude}%C2%B0%20E,{latitude}%C2%B0%20N&name={name}"
}
```

## Fallback Behavior

If no `positionViewer` is configured, the system will automatically fall back to AMap as the default position viewer. This ensures backward compatibility and that GPS location links always work.

## Benefits

This configuration system provides several advantages:
- **Cleaner experience**: Avoid popup-heavy interfaces from some map providers
- **User choice**: Select the map service that works best in your region
- **Flexibility**: Easy to switch between different map providers
- **Backward compatibility**: Existing installations continue to work unchanged