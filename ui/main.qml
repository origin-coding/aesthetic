import QtQuick.Controls.Material
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import QtQuick

ApplicationWindow {
    id: window
    visible: true
    title: "图像美学质量评价系统 v1.0"

    // 窗口最大化且不可更改大小
    visibility:"Maximized"
    flags: Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.Dialog | Qt.WindowTitleHint

    // 默认使用暗色主题
    Material.theme: Material.Dark
    Material.accent: Material.Pink

    // 使用到的属性，统一在这里进行定义
    QtObject {
        id: userInterface

        // 图像的路径
        property string imageSource: ""

        // GitHub仓库的地址
        property string repoURL: "https://github.com/QingyeKiyono/aesthetic"

        // 图像展示区域的宽和高
        property double imageWidth: window.width * 0.4
        property double imageHeight: window.height

        // 结果展示标签的宽和高
        property double resultWidth: 200
        property double resultHeight: 30
    }

    QtObject {
        id: assessResult

        // 图像美学质量高低
        property var /* bool */ binary:             undefined

        // 图像具体美学评分
        property var /* real */ score:              undefined

        // 美学多标签的识别结果，一共有11个
        property var /* bool */ balancingElement:   undefined   // 元素平衡
        property var /* bool */ content:            undefined   // 图像内容
        property var /* bool */ colorHarmony:       undefined   // 色彩和谐
        property var /* bool */ depthOfField:       undefined   // 景深
        property var /* bool */ lighting:           undefined   // 光影
        property var /* bool */ motionBlur:         undefined   // 运动模糊
        property var /* bool */ objectEmphasis:     undefined   // 主题强调
        property var /* bool */ ruleOfThirds:       undefined   // 三分法则
        property var /* bool */ vividColor:         undefined   // 色彩鲜明
        property var /* bool */ repetition:         undefined   // 重复
        property var /* bool */ symmetry:           undefined   // 对称
    }

    // 界面绘制部分相关代码
    menuBar: MenuBar {
        Menu{
            title: "帮助"
            MenuItem {
                text: "关于"
                onTriggered: { aboutDialog.open() }  // 打开关于对话框
            }
            MenuItem {
                text: "源代码"
                onTriggered: { Qt.openUrlExternally(userInterface.repoURL) }  // 打开在线仓库
            }
            MenuItem {
                text: "切换主题"
                onTriggered: { window.Material.theme = window.Material.theme === Material.Light ? Material.Dark : Material.Light }
            }
        }
    }

    // 关于对话框，简单描述应用程序
    Dialog {
        id: aboutDialog
        anchors.centerIn: parent

        modal: true
        standardButtons: Dialog.Ok  // 只有确认选项
        title: "图像美学质量评价系统 v1.0，作者：孙强\n毕业论文《基于深度学习的图像美学质量评价系统设计与实现》界面部分"
    }

    // 文件选择框，选择需要进行评价的图像
    FileDialog {
        id: fileDialog
        title: "选择图像"
        nameFilters: [ "Images (*.png *.jpg)" ]
        onAccepted: {
            userInterface.imageSource = fileDialog.selectedFile
            context.assess_image(userInterface.imageSource)
        }
    }

    RowLayout {
        Item {
            implicitWidth: userInterface.imageWidth; implicitHeight: userInterface.imageHeight

            // 展示图像
            Image {
                width: userInterface.imageWidth; height: userInterface.imageHeight
                fillMode: Image.PreserveAspectFit
                source: userInterface.imageSource
            }
        }

        ColumnLayout {
            RowLayout {
                Button {
                    text: "选择将要识别的图片"
                    onClicked: { fileDialog.open() }
                }
                Label {
                    Layout.leftMargin: 20
                    text: "选择的图片是：" + userInterface.imageSource
                }
            }

            RowLayout {
                Item {
                    implicitWidth: userInterface.resultWidth; implicitHeight: userInterface.resultHeight

                    Label {
                        anchors.verticalCenter: parent
                        text: "识别结果：%1".arg(String(assessResult.binary ?? "unknown"))
                    }
                }

                Item {
                    implicitWidth: userInterface.resultWidth; implicitHeight: userInterface.resultHeight

                    Label {
                        anchors.verticalCenter: parent
                        text: "美学打分：%1".arg(String(assessResult.score ?? "unknown"))
                    }
                }
            }

            RowLayout {
                Item {
                    implicitWidth: userInterface.resultWidth; implicitHeight: userInterface.resultHeight

                    Label {
                        anchors.verticalCenter: parent
                        text: "美学标签-元素平衡：%1".arg(String(assessResult.balancingElement ?? "unknown"))
                    }
                }
                Item {
                    implicitWidth: userInterface.resultWidth; implicitHeight: userInterface.resultHeight

                    Label {
                        anchors.verticalCenter: parent
                        text: "美学标签-图像内容：%1".arg(String(assessResult.content ?? "unknown"))
                    }
                }
                Item {
                    implicitWidth: userInterface.resultWidth; implicitHeight: userInterface.resultHeight

                    Label {
                        anchors.verticalCenter: parent
                        text: "美学标签-色彩和谐：%1".arg(String(assessResult.colorHarmony ?? "unknown"))
                    }
                }

            }

            RowLayout {
                Item {
                    implicitWidth: userInterface.resultWidth; implicitHeight: userInterface.resultHeight

                    Label {
                        anchors.verticalCenter: parent
                        text: "美学标签-景　　深：%1".arg(String(assessResult.depthOfField ?? "unknown"))
                    }
                }
                Item {
                    implicitWidth: userInterface.resultWidth; implicitHeight: userInterface.resultHeight

                    Label {
                        anchors.verticalCenter: parent
                        text: "美学标签-光　　影：%1".arg(String(assessResult.lighting ?? "unknown"))
                    }
                }
                Item {
                    implicitWidth: userInterface.resultWidth; implicitHeight: userInterface.resultHeight

                    Label {
                        anchors.verticalCenter: parent
                        text: "美学标签-运动模糊：%1".arg(String(assessResult.motionBlur ?? "unknown"))
                    }
                }
            }

            RowLayout {
                Item {
                    implicitWidth: userInterface.resultWidth; implicitHeight: userInterface.resultHeight

                    Label {
                        anchors.verticalCenter: parent
                        text: "美学标签-主题强调：%1".arg(String(assessResult.objectEmphasis ?? "unknown"))
                    }
                }
                Item {
                    implicitWidth: userInterface.resultWidth; implicitHeight: userInterface.resultHeight

                    Label {
                        anchors.verticalCenter: parent
                        text: "美学标签-三分法则：%1".arg(String(assessResult.ruleOfThirds ?? "unknown"))
                    }
                }
                Item {
                    implicitWidth: userInterface.resultWidth; implicitHeight: userInterface.resultHeight

                    Label {
                        anchors.verticalCenter: parent
                        text: "美学标签-色彩鲜明：%1".arg(String(assessResult.vividColor ?? "unknown"))
                    }
                }
            }

            RowLayout {
                Item {
                    implicitWidth: userInterface.resultWidth; implicitHeight: userInterface.resultHeight

                    Label {
                        anchors.verticalCenter: parent
                        text: "美学标签-重　　复：%1".arg(String(assessResult.repetition ?? "unknown"))
                    }
                }
                Item {
                    implicitWidth: userInterface.resultWidth; implicitHeight: userInterface.resultHeight

                    Label {
                        anchors.verticalCenter: parent
                        text: "美学标签-对　　称：%1".arg(String(assessResult.symmetry ?? "unknown"))
                    }
                }
            }
        }
    }

    Connections {
        target: context

        // 接收图像评价的结果
        function onSetBinary(result)            { assessResult.binary               = result }
        function onSetScore(result)             { assessResult.score                = result }

        function onSetBalancingElement(result)  { assessResult.balancingElement     = result }
        function onSetContent(result)           { assessResult.content              = result }
        function onSetColorHarmony(result)      { assessResult.colorHarmony         = result }
        function onSetDepthOfField(result)      { assessResult.depthOfField         = result }
        function onSetLighting(result)          { assessResult.lighting             = result }
        function onSetMotionBlur(result)        { assessResult.motionBlur           = result }
        function onSetObjectEmphasis(result)    { assessResult.objectEmphasis       = result }
        function onSetRuleOfThirds(result)      { assessResult.ruleOfThirds         = result }
        function onSetVividColor(result)        { assessResult.vividColor           = result }
        function onSetRepetition(result)        { assessResult.repetition           = result }
        function onSetSymmetry(result)          { assessResult.symmetry             = result }
    }
}
