#pragma once


#include "IO.h"
#include <vector>


thread_local HINSTANCE GhInstance;
thread_local HBITMAP hbmpBack = NULL;



namespace decx
{
    std::vector<std::thread> thr_arr;

    //  窗口消息 处理函数
    static LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

    static int CALLBACK _main(const wchar_t* window_name, uchar* src, int width, int pitch, int height, int channel);


    __declspec(dllexport)
        void _ShowImage(uchar* src, const wchar_t* window_name, int width, int pitch, int height, int channel);


    __declspec(dllexport) void _Wait();
}


static LRESULT CALLBACK decx::WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    HDC hdc;
    HDC HdcMem;

    PAINTSTRUCT ps;

    switch (uMsg)
    {
    case WM_CLOSE:   // press X to close the window
        ::PostQuitMessage(0);  // send the message to quit
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

        SelectObject(hDCMem, hOldBmp);  // restore compatiable DC data
        DeleteDC(hDCMem);   //delete compatiable DC to avoid memory leakage
        EndPaint(hWnd, &ps);
#else
        RECT clientRect;
        GetClientRect(hWnd, &clientRect);

        HDC hdc = GetDC(hWnd);  // get the drawing hdc of the window
        HDC hMdc = CreateCompatibleDC(hdc); // cretae the memory dc
        HBITMAP hMbitmap = (HBITMAP)SelectObject(hMdc, hbmpBack);
#endif
    }
    break;
    }
    return ::DefWindowProc(hWnd, uMsg, wParam, lParam);
}


static int CALLBACK decx::_main(const wchar_t* window_name, uchar *src, int width, int pitch, int height, int channel)
{
    //GhInstance = hInstance;
    // call GDI+ start function  
    ULONG_PTR  ulToken;
    GdiplusStartupInput    gdiplusStartupInput;
    ::GdiplusStartup(&ulToken, &gdiplusStartupInput, NULL);

    Bitmap* lpbmp = new Bitmap(width, height);
    if (channel == 4) {
        decx::TranslateMyCls_4(src, lpbmp, pitch);       // load the pixels in img to Bitmap for displaying buffer
    }
    else if (channel == 1) {
        decx::TranslateMyCls_1(src, lpbmp, pitch);
    }

    lpbmp->GetHBITMAP(0, &hbmpBack);    // get HBITMAP 

    //  1.  设计
    WNDCLASSEX wndclass;
    wndclass.cbClsExtra = 0;
    wndclass.cbWndExtra = 0;   //  extra space?
    wndclass.cbSize = sizeof(WNDCLASSEX);
    wndclass.hbrBackground = NULL;   //  default background color
    wndclass.hCursor = NULL; // default cursor
    wndclass.hIcon = NULL;  //default pattern
    wndclass.hIconSm = NULL;  //  small pattern on left and upper side of the window
    wndclass.hInstance = GhInstance;    //  current handle
    wndclass.lpfnWndProc = WndProc;    //  the function that processes the messages
    wndclass.lpszClassName = (LPCWSTR)window_name;   //  name of the registered window
    wndclass.lpszMenuName = NULL;       //  name of the menu
    wndclass.style = CS_HREDRAW | CS_VREDRAW;  //  pattern of the window

    //  2.  register
    if (::RegisterClassEx(&wndclass) == FALSE){
        ::MessageBox(NULL, (LPCWSTR)L"Failed to register", (LPCWSTR)"Error", MB_OK);
        return 0;
    }

    //  3.  create
    // x, y, nW, nH
    HWND hWnd = ::CreateWindow(wndclass.lpszClassName, (LPCWSTR)window_name, WS_OVERLAPPEDWINDOW, 100, 0,
        lpbmp->GetWidth(), lpbmp->GetHeight(), NULL, NULL, GhInstance, NULL);

    if (hWnd == NULL) {
        ::MessageBox(NULL, (LPCWSTR)"Failed to create window", (LPCWSTR)"Error", MB_OK);
        return 0;
    }
    //  4.   display
    ::ShowWindow(hWnd, SW_SHOW);
    //  5.   message main-loop
    MSG msg;
    while (::GetMessage(&msg, 0, 0, 0))
    {
        //  translate
        ::TranslateMessage(&msg);
        //  distribute
        ::DispatchMessage(&msg);
    }
    // close GDI+   
    GdiplusShutdown(ulToken);
    return 0;
}





void decx::_ShowImage(uchar* src, const wchar_t* window_name, int width, int pitch, int height, int channel)
{
    decx::thr_arr.push_back(
        std::thread(decx::_main, window_name, src, width, pitch, height, channel)
    );
}



void decx::_Wait()
{
    for (int i = 0; i < decx::thr_arr.size(); ++i) {
        decx::thr_arr[i].join();
    }
}