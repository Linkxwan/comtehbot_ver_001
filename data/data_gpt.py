# список c "ответами"
texts = [
    'полное название колледжа БИШКЕКСКИЙ КОЛЛЕДЖ КОМПЬЮТЕРНЫХ СИСТЕМ И ТЕХНОЛОГИЙ',
    'специальности колледжа комтехно Студенты могут изучать программное обеспечение вычислительной техники и автоматизированных систем, техническое обслуживание средств вычислительной техники и компьютерных сетей, прикладную информатику в разных областях, дизайн в различных отраслях, экономику и бухгалтерский учет, менеджмент, банковское дело. Учебная программа первого курса соответствует программе 10-11 классов.',
    'Бегалиев Улугбек Турдалиевич - Директор к.т.н., член Международных Ассоциаций и Общества по сейсмостойкому строительству и антисейсмическим системам EERI (USA), ASSISI (Italy). \n\nДавлетбекова Айзада Давлетбековна - Заместитель директора по УВР,  отличник образования Кыргызской Республике. \n\nАширбаева Эльмира Малабековна - Менеджер отдела обеспечения и контроля качества. \n\nКенешбаева Майрам Мукангалиевна - Председатель “Учебно-методического совета” кандидат экономических наук. \n\nИсхакова Гульбахар Ашимжановна - Зав. отделения “Информатики, вычислительной техники и дизайна”, старший преподаватель.\n\nАлтыбаева Шааркуль Исаковна - Зав. отделения “Экономики, бухгалтерского учёта и банковского дела”, старший преподаватель.',
    'Какой адрес у колледжа по какому адресу находитесь Адрес - г. Бишкек, ул. Анкара (Горького), 1/17.  Контактный номер WhatsApp по номеру контакты 0707 37 99 57',
    'инстаграм instagram колледжа @comtehno.kg. фейсбук facebook колледжа facebook.com/contehno.kg. сайт колледжа по адресу www.comtehno.kg',
    'Как можно зачислиться в колледж Каким образом можно поступить в колледж Опиши процесс поступления в колледж 1.Вы делаете онлайн регистрацию 2.Затем вы подаёте заявление на поступление 3.Далее вы подаёте необходимые документы 4.Оплачиваете обучение 5.Вы студент "КОМТЕХНО".'
    'Какое расписание у студентов по учебе в день в какое время начинаются занятия Если вы на первой смене с 8.00 до 14.00, если вторая смена с 11.00 до 17.00. ',
    'Срок обучения в колледже составляет: Нормативный срок обучения: 2 года 10 месяцев (после 9 класса); Нормативный срок обучения: 1 года 10 месяцев (после 11 класса);',
    'В колледж «КОМТЕХНО» принимаем абитуриентов: на базе 9 класса на первый курс; на базе 11 класса на второй курс.  Документ об образовании, выдаваемый по завершению образования: диплом государственного образца. ',
    'перечень документов необходимых для поступления При личном визите для поступления в КОМТЕХНО абитуриенты предоставляют следующие документы: 1.фото 3х4 см. – 6 шт.; 2.копия паспорта или свидетельства о рождении; 3.оригинал свидетельства/аттестата об общем образовании; При электронной регистрации для поступления в КОМТЕХНО абитуриенты предоставляют следующие документы: 1.электронная версия фотографии размером 3х4 см; 2.скан-копия паспорта или свидетельства о рождении; 3.скан-копия свидетельства/аттестата об общем образовании;',
    'Хотел узнать расписание заочников Здравствуйте, расписание у дистанционного (заочного) обучение вы можете увидеть на официальном сайте МУИТ, в разделе расписания.',
    'Пароль и логин от портала moodle вы получите пройдя по ссылке moodle.intuit.kg и позвонив по указанному номеру администратора Эркинбек кызы Гулиза',
    'По поводу дистанционного заочного обучения ответственная Эркинбек кызы Гулиза, она находится на 3-этаже в 346-кабинете.',
    'как проходит сессия что такое сессия Сессия – период сдачи экзаменов за текущий семестр. Она состоит из нескольких экзаменов. Порядок и правила сдачи похожи на сдачу школьных экзаменов. Из отметки, полученной за экзамен и отметки, полученной за две модульные недели, выводится общий итог по предмету. Именно этот балл будет выставлен на ведомость.',
    'Что такое модульная неделя Модульная неделя – это такая же неделя с обычным расписанием занятий. Только преподаватели в течении нее проводят контрольные, подводят некоторые итоги лабораторных занятий для промежуточного контроля знаний. Таких модульных недель в учебном семестре 2. Итоги двух модульных недель могут повлиять на отметку, с которой вы выходите на экзамен или зачет. Что такое числительные и знаменательные недели почему расписание такое сложное числитель и знаменатель недели Расписание на первый взгляд кажется непонятным. На самом деле в нем очень легко разобраться. Существуют белые и черные недели, которые идут друг за другом. А отличие между этими неделями заключается в занятиях. Многие пары постоянны как на белой, так и на черной неделе. А есть пары, которые меняются или вовсе исчезают от недели к неделе. За частую это лабораторные и практические занятия.',
    'Где найти расписание можно посмотреть на сайте университета. Перейдя на него, нажав в правом нижнем углу «Расписание», откроется вкладка с главными ссылками по университету. Здесь же, для удобства, можно сразу перейти на вкладку с расписанием и найти пары именно вашей группы',
    'С какого курса мы начнём делать курсовую работу на компьютере Со 2 курса по результатам специальных дисциплин у вас будут навыки для разработки курсового проекта.',
    'Где можно получить информацию о колледже Вы можете получить необходимую информацию на сайте www.comtehno.kg',
]