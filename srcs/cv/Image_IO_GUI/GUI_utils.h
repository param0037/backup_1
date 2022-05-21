#pragma once


#include "IO.h"
#include <vector>


//GDI+绘图
//准备：将图片放到与代码的同一级目录下

thread_local HINSTANCE GhInstance;
thread_local HBITMAP hbmpBack = NULL;


std::vector<std::thread> thr_arr;



__declspec(dllexport)
void GDIPlusInit()
{
    ULONG_PTR  ulToken;
    GdiplusStartupInput    gdiplusStartupInput;
    ::GdiplusStartup(&ulToken, &gdiplusStartupInput, NULL);
}



static
//  窗口消息 处理函数
LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    HDC hdc;
    HDC HdcMem;

    PAINTSTRUCT ps;

    switch (uMsg)
    {
    case WM_CLOSE:   //  点X 窗口关闭的消息
        ::PostQuitMessage(0);  //  发送退出的消息
        break;
    case WM_PAINT:
    {
#ifndef BitMapOnly
        hdc = BeginPaint(hWnd, &ps);
        HDC hDCMem = CreateCompatibleDC(hdc);
        HBITMAP hOldBmp = (HBITMAP)SelectObject(hDCMem, hbmpBack);

        BITMAP bmp;
        GetObject(hbmpBack, sizeof(BITMAP), &bmp);

        BitBlt(hdc, 0, 0, bmp.bmWidth, bmp.bmHeight, hDCMem, 0, 0, SRCCOPY);

        SelectObject(hDCMem, hOldBmp);//复原兼容DC数据
        DeleteDC(hDCMem);//删除兼容DC，避免内存泄漏
        EndPaint(hWnd, &ps);
#else
        RECT clientRect;
        GetClientRect(hWnd, &clientRect);

        HDC hdc = GetDC(hWnd);//获取窗口的绘图hdc
        HDC hMdc = CreateCompatibleDC(hdc);//创建内存dc
        HBITMAP hMbitmap = (HBITMAP)SelectObject(hMdc, hbmpBack);
#endif
    }
    break;
    }
    return ::DefWindowProc(hWnd, uMsg, wParam, lParam);
}


static
int CALLBACK _main(const wchar_t* window_name, uchar *src, int width, int height, int channel)
{
    //GhInstance = hInstance;
    //调用GDI+开始函数  
    ULONG_PTR  ulToken;
    GdiplusStartupInput    gdiplusStartupInput;
    ::GdiplusStartup(&ulToken, &gdiplusStartupInput, NULL);

    Bitmap* lpbmp = new Bitmap(width, height);
    if (channel == 4) {
        TranslateMyCls_4(src, lpbmp);       // 把 img 中的像素值加载进 Bitmap 类里作为显示用 buffer
    }
    else if (channel == 1) {
        TranslateMyCls_1(src, lpbmp);
    }

    lpbmp->GetHBITMAP(0, &hbmpBack);    // 得到 HBITMAP 

    //  1.  设计
    WNDCLASSEX wndclass;
    wndclass.cbClsExtra = 0;
    wndclass.cbWndExtra = 0;   //  是否要分配额外的空间
    wndclass.cbSize = sizeof(WNDCLASSEX);
    wndclass.hbrBackground = NULL;   //  背景颜色默认
    wndclass.hCursor = NULL; // 默认光标
    wndclass.hIcon = NULL;  //默认图标
    wndclass.hIconSm = NULL;  //  窗口左上小图标
    wndclass.hInstance = GhInstance;    //  当前实例的句柄
    wndclass.lpfnWndProc = WndProc;    //  消息处理函数
    wndclass.lpszClassName = (LPCWSTR)window_name;   //  注册窗口类的名
    wndclass.lpszMenuName = NULL;       //  菜单名
    wndclass.style = CS_HREDRAW | CS_VREDRAW;  //  窗口类的样式

    //  2.  注册
    if (::RegisterClassEx(&wndclass) == FALSE)
    {
        ::MessageBox(NULL, (LPCWSTR)L"注册失败", (LPCWSTR)"提示", MB_OK);
        return 0;
    }

    //  3.  创建
    // x, y, nW, nH
    HWND hWnd = ::CreateWindow(wndclass.lpszClassName, (LPCWSTR)window_name, WS_OVERLAPPEDWINDOW, 100, 0,
        lpbmp->GetWidth(), lpbmp->GetHeight(), NULL, NULL, GhInstance, NULL);

    if (hWnd == NULL)
    {
        ::MessageBox(NULL, (LPCWSTR)"创建失败", (LPCWSTR)"提示", MB_OK);
        return 0;
    }
    //  4.   显示
    ::ShowWindow(hWnd, SW_SHOW);
    //  5.   消息循环
    MSG msg;
    while (::GetMessage(&msg, 0, 0, 0))
    {
        //  翻译
        ::TranslateMessage(&msg);
        //  分发
        ::DispatchMessage(&msg);
    }
    //关闭GDI+   
    GdiplusShutdown(ulToken);
    return 0;
}




// This DLL
__declspec(dllexport)
void _ShowImage(uchar* src, const wchar_t* window_name, int width, int height, int channel)
{
    thr_arr.push_back(
        std::thread(_main, window_name, src, width, height, channel)
    );
}


__declspec(dllexport)
void _Wait()
{
    for (int i = 0; i < thr_arr.size(); ++i) {
        thr_arr[i].join();
    }
}